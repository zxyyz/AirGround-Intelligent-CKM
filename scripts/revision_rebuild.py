"""E3 fresh measurements from physical rays; learner data excludes ray metadata."""
import argparse,json,time,resource,os
from pathlib import Path
import numpy as np
import torch
from paths import EXP,config,save,sha,stamp,seed
from data import freeze_queries,folds
from physics import SignalWorld,fold
from collector import Receiver
from ckm_sim.sounding import simulate_received
from ckm_sim.metrics import squared_gospa
from revision_frontend import synth
from revision_metrics import csvout
OLD=EXP.parent/'geometry_feature_attribution_20261027'

def setup():
 assert json.loads((EXP/'results/frontend_pilot_summary.json').read_text())['passed']
 if not (EXP/'cache/pool.npz').exists():
  src=np.load(OLD/'cache/pool.npz');mask=src['train'];p={k:src[k][mask] if k!='unique' else src[k] for k in src.files}
  p['labels']=np.zeros_like(p['labels']);p['n_paths']=np.zeros_like(p['n_paths']);p['valid']=np.zeros_like(p['valid']);p['train']=np.zeros_like(p['train'])
  # Candidate eligibility is position/ID-only, fixed before the new measurements.
  p['candidate']=np.ones(len(p['x']),bool)
  np.savez_compressed(EXP/'cache/pool.npz',**p)
  np.savez_compressed(EXP/'cache/pool_positions_frozen.npz',**{k:p[k] for k in ['x','ids','pair','unique','unique_index','fold','buffer_excluded','candidate']})
 # freeze_queries expects the eligible training positions before validity is known.
 if not (EXP/'cache/queries_confirm.npz').exists():
  import data
  src=np.load(EXP/'cache/pool.npz');p={k:src[k] for k in src.files};p['train']=p['candidate'];np.savez_compressed(EXP/'cache/pool.npz',**p);freeze_queries();p['train'][:]=False;np.savez_compressed(EXP/'cache/pool.npz',**p)
 q=np.load(EXP/'cache/queries_confirm.npz')['queries'];oldq=np.load(OLD/'cache/queries_confirm.npz')['queries'];assert not(set(map(tuple,q))&set(map(tuple,oldq)))
 save(EXP/'audit/confirmation_boundary.json',dict(seed=2026102902,n=4096,query_sha256=sha(EXP/'cache/queries_confirm.npz'),old_confirm_overlap=0,new_reference_exists=(EXP/'cache/reference_confirm.npz').exists(),role='Public coordinates frozen before development; no reference/score generated until hyperparameter freeze',created_at=stamp()))
 assert (EXP/'configs/query_freeze_dev.json').exists(), 'Supply the archived development-coordinate freeze before rebuilding.'

def generate_pool_rays():
 pos=np.load(EXP/'cache/pool_positions_frozen.npz');unique=pos['unique'];uids=np.unique(pos['unique_index']);out=EXP/'cache/physical_pool';out.mkdir(exist_ok=True)
 world=SignalWorld('W1a',config());save(EXP/'audit/new_pool_ray_world.json',world.audit.manifest);start=time.perf_counter();count=0
 for uid in uids:
  path=out/f'{uid:05d}.json'
  if path.exists():continue
  _,rec,_,_=world.channel(unique[uid]);rec['unique_position_id']=int(uid);save(path,rec);count+=1
  if count%128==0:print('RAYS',count,len(uids),round(time.perf_counter()-start,1),flush=True)
 save(EXP/'results/new_pool_ray_resources.json',dict(unique_positions=len(uids),new_traces_this_run=count,wall_seconds=time.perf_counter()-start,source='new Sionna RT from unchanged physical scene and positions',created_at=stamp()))

def pool_labels(shard=0,shards=1):
 c=config();batch=int(os.environ.get('REV_RF_BATCH','32'));c['runtime']['receiver_batch_size']=batch;recvr=Receiver(c);src=np.load(EXP/'cache/pool_positions_frozen.npz');n=len(src['x']);start=time.perf_counter();out=EXP/'cache/measurement_parts';out.mkdir(exist_ok=True);records={};new=0;failures=[]
 for first in range(shard*128,n,shards*128):
  path=out/f'{first:06d}.npz'
  if path.exists():continue
  end=min(first+128,n);zs=[];qs=[];times=[]
  for sub in range(first,end,batch):
   signals=[]
   for i in range(sub,min(end,sub+batch)):
    uid=int(src['unique_index'][i]);rec=records.get(uid)
    if rec is None:rec=json.loads((EXP/f'cache/physical_pool/{uid:05d}.json').read_text());records[uid]=rec
    h,_=synth(rec,c);signal,p=simulate_received(h,(20261029,int(src['pair'][i]),seed('measurement',str(src['ids'][i]))),c);signals.append(signal)
   tic=time.perf_counter();z,q=recvr.estimate(np.stack(signals),p);torch.cuda.synchronize();times.extend([(time.perf_counter()-tic)/len(z)]*len(z));zs.extend(z);qs.extend(q)
  labels=np.zeros((end-first,3,6));k=np.array([len(z) for z in zs]);valid=np.array([q['valid'] for q in qs])
  for j,z in enumerate(zs):labels[j,:len(z)]=z
  for j,q in enumerate(qs):
   if not q['valid']:failures.append(dict(index=first+j,id=str(src['ids'][first+j]),quality=q))
  tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,indices=np.arange(first,end),labels=labels,n_paths=k,valid=valid,extraction_seconds=times);tmp.replace(path)
  save(path.with_suffix('.json'),dict(first=first,end=end,quality=qs,protocol_sha256=sha(EXP/'configs/effective_config.json')))
  new+=end-first
  if first%2048==0:print('MEASUREMENTS',end,n,round(time.perf_counter()-start,1),flush=True)
 save(EXP/f'results/new_measurement_resources_{shard}.json',dict(shard=shard,shards=shards,new_extractions=new,wall_seconds=time.perf_counter()-start,device=torch.cuda.get_device_name(),peak_memory_bytes=torch.cuda.max_memory_allocated(),failures=failures,created_at=stamp()))

def assemble_pool():
 src=np.load(EXP/'cache/pool_positions_frozen.npz');n=len(src['x']);out=EXP/'cache/measurement_parts';start=time.perf_counter();new=0
 parts=[np.load(out/f'{first:06d}.npz') for first in range(0,n,128)];p={k:src[k] for k in src.files}
 for k in ['labels','n_paths','valid']:p[k]=np.concatenate([a[k] for a in parts])
 failed_suffix={str(i).split('/',2)[2] for i,ok in zip(p['ids'],p['valid']) if not ok};shared_valid=np.array([str(i).split('/',2)[2] not in failed_suffix for i in p['ids']]);p['train']=p['valid']&shared_valid
 assert np.isfinite(p['labels']).all();counts={str(f):int(np.sum(p['train']&(p['pair']==f))) for f in range(5101,5111)};assert len(set(counts.values()))==1
 np.savez_compressed(EXP/'cache/pool.npz',**p)
 save(EXP/'results/rebuilt_pool.json',dict(records=n,valid=int(p['valid'].sum()),training=int(p['train'].sum()),by_flow=counts,failed_ids=[str(i) for i,ok in zip(p['ids'],p['valid']) if not ok],excluded_common_id_suffixes=sorted(failed_suffix),empty_training=int(np.sum(p['train']&(p['n_paths']==0))),new_extractions_this_run=new,wall_seconds=time.perf_counter()-start,extraction_seconds_sum=float(sum(a['extraction_seconds'].sum() for a in parts)),pool_sha256=sha(EXP/'cache/pool.npz'),created_at=stamp()))

def dev_reference():
 dest=EXP/'cache/reference_dev.npz'
 if dest.exists():return
 c=config();receiver=Receiver(c);q=np.load(EXP/'cache/queries_dev.npz')['queries'];records=[]
 for p in sorted((EXP/'audit/reproduction_inputs/development_rays').glob('dev_*_rt.json')):records.extend(json.loads(p.read_text()))
 assert len(q)==len(records);labels=[];masks=[];counts=[];start=time.perf_counter();qualities=[]
 for first in range(0,len(q),8):
  signals=[];physical=[]
  for i in range(first,min(first+8,len(q))):
   np.testing.assert_array_equal(q[i],records[i]['position']);h,phys=synth(records[i],c);s,p=simulate_received(h,(2026102681,0,i),c,noiseless=True);signals.append(s);physical.append(phys)
  zz,qq=receiver.estimate(np.stack(signals),p);assert all(x['valid'] for x in qq);qualities.extend(qq)
  for j,z in enumerate(zz):
   rec=records[first+j];ix=[i for i in rec['signal_indices'] if rec['paths'][i]['z'][0]>=-110][:3];_,pairs,_,_=squared_gospa(z,physical[j],c);mask=np.zeros(3,bool)
   for a,b in pairs:mask[a]=rec['paths'][ix[b]]['kind']=='1+1'
   labels.append(z);masks.append(mask);counts.append([len(ix),sum(rec['paths'][i]['kind']=='1+1' for i in ix)])
 k=np.array([len(z) for z in labels]);z=np.zeros((len(k),3,6))
 for i,a in enumerate(labels):z[i,:len(a)]=a
 np.savez_compressed(dest,reference=z,n_paths=k,observable_second_mask=masks,physical_counts=counts,weights=np.full(len(k),1/len(k)))
 save(EXP/'results/rebuilt_development.json',dict(queries=len(k),counts=np.bincount(k,minlength=4),failures=0,quality=qualities,wall_seconds=time.perf_counter()-start,reference_sha256=sha(dest),created_at=stamp()))

if __name__=='__main__':
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['setup','rays','measurements','assemble','dev','all']);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);a=ap.parse_args()
 if a.stage=='all':setup();generate_pool_rays();pool_labels();assemble_pool();dev_reference()
 elif a.stage=='measurements':pool_labels(a.shard,a.shards)
 else:{'setup':setup,'rays':generate_pool_rays,'assemble':assemble_pool,'dev':dev_reference}[a.stage]()
