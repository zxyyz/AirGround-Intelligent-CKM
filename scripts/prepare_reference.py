"""Evaluation-only Sionna -> noiseless sounding -> frozen MUSIC/LS reference."""
import argparse, json, time
import numpy as np
import torch
from paths import EXP, config, save, sha, stamp
from physics import SignalWorld, fold
from collector import Receiver
from ckm_sim.sounding import simulate_received
from ckm_sim.metrics import squared_gospa


def run(name,shard,shards,limit=0):
    assert json.loads((EXP/'configs/freeze.json').read_text())['frozen']
    torch.set_num_threads(2)
    cfg=config();cfg['runtime']['device']='cuda:0'
    qfile=EXP/f'cache/queries_{name}.npz';q=np.load(qfile)['queries']
    frozen=json.loads((EXP/f'configs/query_freeze_{name}.json').read_text())
    assert frozen['sha256']==sha(qfile)
    world=SignalWorld('W1a',cfg);receiver=Receiver(cfg)
    save(EXP/f'results/reference_world_{name}_{shard}.json',world.audit.manifest)
    start=time.time();done=0
    for first in range(shard*64,len(q),shards*64):
        out=EXP/f'cache/reference_parts/{name}_{first:05d}.npz'
        if out.exists() and out.with_name(out.stem+'_rt.json').exists():continue
        end=min(len(q),first+64);labels=[];masks=[];pcounts=[];quality=[];rt_records=[]
        for sub in range(first,end,8):
            signals=[];records=[]
            for i in range(sub,min(end,sub+8)):
                h,rec,_,_=world.channel(q[i]);s,p=simulate_received(h,(frozen['seed'],0,i),cfg,noiseless=True)
                signals.append(s);records.append(rec)
            zs,qs=receiver.estimate(np.stack(signals),p)
            for z,qual,r in zip(zs,qs,records):
                if not qual['valid']:raise RuntimeError(f'Invalid reference {name} {first}: {qual}')
                labels.append(z);quality.append(qual)
                ix=[j for j in r['signal_indices'] if r['paths'][j]['z'][0]>=-110][:3]
                physical=fold(np.array([r['paths'][j]['z'] for j in ix]).reshape(-1,6))
                _,pairs,_,_=squared_gospa(np.asarray(z,dtype=np.float64),physical,cfg)
                mask=np.zeros(3,bool)
                for oi,pi in pairs:mask[oi]=r['paths'][ix[pi]]['kind']=='1+1'
                masks.append(mask);pcounts.append([len(ix),sum(r['paths'][j]['kind']=='1+1' for j in ix)])
                rt_records.append(r)
        k=np.array([len(z) for z in labels],np.int32);z=np.zeros((len(k),3,6))
        for j,a in enumerate(labels):z[j,:len(a)]=a
        out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out,reference=z,n_paths=k,observable_second_mask=np.array(masks),physical_counts=np.array(pcounts),indices=np.arange(first,end))
        # Physical metadata is confined to evaluation cache; never opened by training.
        save(EXP/f'cache/reference_parts/{name}_{first:05d}_rt.json',rt_records)
        done+=len(k); print('REFERENCE',name,shard,first,end,round(time.time()-start,1),flush=True)
        if limit and done>=limit:break


def assemble(name):
    q=np.load(EXP/f'cache/queries_{name}.npz')['queries'];parts=[]
    for first in range(0,len(q),64):
        p=EXP/f'cache/reference_parts/{name}_{first:05d}.npz'
        if not p.exists():return False
        parts.append(np.load(p))
    arrays={k:np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    assert np.array_equal(arrays.pop('indices'),np.arange(len(q)))
    out=EXP/f'cache/reference_{name}.npz';np.savez_compressed(out,**arrays,weights=np.full(len(q),1/len(q)))
    save(EXP/f'results/reference_{name}.json',dict(n=len(q),counts=np.bincount(arrays['n_paths'],minlength=4).tolist(),sha256=sha(out),created_at=stamp(),source='unchanged W1a Sionna RT 2.2.0; noiseless sounding; frozen MUSIC-LS; float64 evaluation',second_components=int(arrays['observable_second_mask'].sum())))
    return True


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name',choices=['dev','confirm','domain']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--limit',type=int,default=0);p.add_argument('--assemble',action='store_true');a=p.parse_args()
    if not a.assemble:run(a.name,a.shard,a.shards,a.limit)
    print('ASSEMBLED',a.name,assemble(a.name),flush=True)
