"""Formal trajectory/learning worker. No evaluator or test-query access."""
from common import *
import argparse,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np,torch
from formal_models import make_model,Features
from collector import worker_init,observe,Receiver,collect
from store import history,get_window,has_window
from coverage import plan
from sampling_log import trace
from ckm_sim.v14_planning import initial_state,independent_segment
from ckm_sim.geometry import GeometryMap
from ckm_sim.experiment import weight_hash

def checkpoint(path,m,state,r,stream,timing,training_log,record_digest):
    s=dict(committee=copy.deepcopy(m.state_dict()),state=copy.deepcopy(state),round=r,stream=stream,n_attempts=1800*(r+1),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),weight_hash=weight_hash(m),variant=m.variant,timing=timing,training_log=training_log,record_digest=record_digest)
    if m.variant=='D':s['initialization_namespace']='D-replay:init:SHA256-63bit:v2'
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(s,tmp);tmp.replace(path)
    return s

def restore(m,s):
    assert m.variant==s['variant'];m.load_state_dict(copy.deepcopy(s['committee']));torch.set_rng_state(s['torch_rng'].cpu());torch.cuda.set_rng_state_all([a.cpu() for a in s['cuda_rng']]);assert weight_hash(m)==s['weight_hash']
    state=copy.deepcopy(s['state']);state['x']=np.array(state['x']);return state

def train(path,m,g,phase,pair,stream,r,meta,digest):
    check_deadline()
    path.parent.mkdir(parents=True,exist_ok=True)
    x,z,k,valid=history(phase,pair,stream,r);poses=x[valid];labels=[a[:n] for a,n in zip(z[valid],k[valid])];m.poses=poses
    before=time.perf_counter();features=g.features(poses);geom=time.perf_counter()-before;steps=1500 if r==0 else 800
    batchfile=path.parent/f'sampling_{r:02d}.npz';hashes=trace(batchfile,m,poses,k[valid],steps,np.flatnonzero(valid))
    before=time.perf_counter();logs=m.train(features,labels,steps);training=time.perf_counter()-before
    assert all(np.isfinite(a['gradient_norm_mean']) and np.isfinite(a['loss_end']) for a in logs)
    assert hashes==[a['sample_index_sha256'] for a in logs];assert m.steps==[1500+800*r]*3
    timing=dict(**meta['timing'],model_geometry_s=geom,training_s=training)
    s=checkpoint(path,m,meta['state'],r,stream,timing,logs,digest)
    save(path.with_suffix('.json'),dict(round=r,variant=m.variant,stream=stream,steps=m.steps,n_attempts=len(x),n_valid=int(valid.sum()),weight_hash=s['weight_hash'],sampling_sha256=hashes,timing=timing))
    print('TRAINED',phase,pair,path.parent.parent.name,r,m.steps[0],flush=True);return s

def assert_frozen():
    f=json.loads((EXP/'configs/freeze.json').read_text())
    # Evaluation-cache hashes are verified by the launcher, not this learner.
    for p,v in f['hashes'].items():
        if '/cache/' not in p:assert sha(ROOT/p)==v,('frozen file changed',p)
    return f

def run(pair,phase):
    start=time.perf_counter();cfg=config();windows=2 if phase=='smoke' else 8
    if phase=='formal':assert pair in assert_frozen()['seeds']
    else:assert pair==5099
    torch.set_num_threads(2);g=GeometryMap(cfg);receiver=Receiver(cfg);root=EXP/f'runs/{phase}/{pair}';root.mkdir(parents=True,exist_ok=True)
    if (root/'complete.json').exists():return
    save(root/'worker.json',dict(pair=pair,phase=phase,device=torch.cuda.get_device_name(),visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),pid=os.getpid(),start_epoch=time.time(),query_or_reference_access=False))
    with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn'),initializer=worker_init,initargs=(cfg,)) as pool:
        if not has_window(phase,pair,'I',0):
            state=initial_state(cfg);before=time.perf_counter();actions,state,log=independent_segment(state,g,cfg,1800,1800)
            collect(phase,pair,'I',0,actions,state,log,time.perf_counter()-before,g,receiver,pool,cfg)
        _,meta,digest=get_window(phase,pair,'I',0)
        fg=Features(cfg,'M3');fd=Features(cfg,'D');initial={}
        for variant,feat in [('D',fd),('M3',fg)]:
            dest=root/f'initial_{variant}/checkpoints/update_00.pt';m=make_model(cfg,pair,variant)
            if dest.exists():initial[variant]=torch.load(dest,map_location=cfg['runtime']['device'],weights_only=False)
            else:initial[variant]=train(dest,m,feat,phase,pair,'I',0,meta,digest)
        assert [a['sample_index_sha256'] for a in initial['D']['training_log']]==[a['sample_index_sha256'] for a in initial['M3']['training_log']]
        for stream,branches in [('I',[('I-D','D',fd),('I-G','M3',fg)]),('C',[('C-G','M3',fg)]),('P',[('P-G','M3',fg)])]:
            models={};states={}
            for name,variant,features in branches:
                dest=root/name/'checkpoints';dest.mkdir(parents=True,exist_ok=True);zero=dest/'update_00.pt'
                if not zero.exists():os.link(root/f'initial_{variant}/checkpoints/update_00.pt',zero)
                m=make_model(cfg,pair,variant);models[name]=m;states[name]=restore(m,initial[variant])
                save(root/name/'manifest.json',dict(status='running',pair=pair,branch=name,variant=variant,stream=stream,initial_weight_hash=initial[variant]['weight_hash'],initial_record_digest=digest,threshold=.5,shared_initial_deep_clone=True,expected_windows=windows))
            for r in range(1,windows):
                check_deadline()
                existing={name:root/name/f'checkpoints/update_{r:02d}.pt' for name,_,_ in branches}
                if all(p.exists() for p in existing.values()):
                    for name,m in models.items():states[name]=restore(m,torch.load(existing[name],map_location=m.device,weights_only=False))
                    continue
                if not has_window(phase,pair,stream,r):
                    previous=root/branches[-1][0]/f'checkpoints/update_{r-1:02d}.pt'
                    m=models[branches[-1][0]];state=restore(m,torch.load(previous,map_location=m.device,weights_only=False));before=time.perf_counter()
                    if stream=='I':actions,state,log=independent_segment(state,g,cfg,1800,1800)
                    else:
                        hx,_,_,_=history(phase,pair,stream,r-1)
                        actions,state,log,candidates=plan('CG' if stream=='C' else 'P0',m,fg,hx,state,cfg,pair*10000+r,{})
                        np.savez_compressed(root/branches[0][0]/f'candidates_{r:02d}.npz',**candidates)
                    planning=time.perf_counter()-before
                    collect(phase,pair,stream,r,actions,state,log,planning,g,receiver,pool,cfg)
                _,meta,rd=get_window(phase,pair,stream,r)
                for name,variant,features in branches:
                    m=models[name];p=existing[name]
                    if p.exists():states[name]=restore(m,torch.load(p,map_location=m.device,weights_only=False));continue
                    prior=root/name/f'checkpoints/update_{r-1:02d}.pt';restore(m,torch.load(prior,map_location=m.device,weights_only=False))
                    saved=train(p,m,features,phase,pair,stream,r,meta,rd);states[name]=saved['state']
                if stream=='I':
                    a=torch.load(existing['I-D'],map_location='cpu',weights_only=False);b=torch.load(existing['I-G'],map_location='cpu',weights_only=False)
                    assert [x['sample_index_sha256'] for x in a['training_log']]==[x['sample_index_sha256'] for x in b['training_log']]
            for name,variant,features in branches:
                final=torch.load(root/name/f'checkpoints/update_{windows-1:02d}.pt',map_location='cpu',weights_only=False)
                assert final['committee']['steps']==[1500+800*(windows-1)]*3
                save(root/name/'manifest.json',dict(status='complete',pair=pair,branch=name,variant=variant,stream=stream,initial_weight_hash=initial[variant]['weight_hash'],initial_record_digest=digest,threshold=.5,shared_initial_deep_clone=True,windows=windows,attempts=1800*windows,steps=final['committee']['steps'],total_service_s=windows*427.96875,physical_total_s=windows*(1800+427.96875)))
        save(root/'geometry_cost.json',dict(unique_positions=len(fg.cache),candidate_counts=fg.costs))
    save(root/'complete.json',dict(status='complete',phase=phase,pair=pair,wall_s=time.perf_counter()-start,epoch=time.time(),peak_torch_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',type=int,required=True);p.add_argument('--phase',choices=['smoke','formal'],required=True);a=p.parse_args()
    try:run(a.pair,a.phase)
    except Exception as e:failure('runner',e);raise
