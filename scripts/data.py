"""Public inputs and fixed noisy training data. Never loads evaluation labels."""
import argparse,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from paths import EXP,config,seed,save,sha,stamp
from ckm_sim.geometry import GeometryMap
WIDTH=np.array([60,60,30,60,60])
def blocks(x):return np.floor(np.asarray(x)[...,:5]/WIDTH).astype(np.int16)
def folds(x):
    import hashlib
    return np.array([int.from_bytes(hashlib.sha256(('geometry-set-ckm-split-v1:'+':'.join(map(str,b))).encode()).digest()[:8],'big')%5 for b in blocks(x)],np.int8)
def subset_indices(p,pair,ratio):
    if not isinstance(p,dict):p={k:p[k] for k in p.files}
    indices=np.flatnonzero(p['train']&(p['pair']==pair));groups={}
    for i,b in zip(indices,blocks(p['x'][indices])):groups.setdefault((*b,int(p['n_paths'][i])),[]).append(int(i))
    selected=[]
    # Keep the original sampling namespace so frozen nested pools are reused exactly.
    import hashlib
    def old_seed(frame):return int.from_bytes(hashlib.sha256(('geometry-set-ckm:subset:'+str(frame)).encode()).digest()[:8],'big')%(2**63-1)
    for v in groups.values():
        v.sort(key=lambda i:old_seed(p['ids'][i]));selected.extend(v[:int(np.ceil(ratio*len(v)))])
    return np.array(sorted(selected),np.int32)
def raw_map(x,buildings=None):
    cfg=config();g=GeometryMap(cfg);b=g.buildings if buildings is None else np.asarray(buildings)
    assert b.shape==(16,5)
    x=np.atleast_2d(x);scale=np.array([240.,240.,120.]);pose=x/np.tile(scale,2)
    lo=np.c_[b[:,:2],np.zeros(16)];hi=np.c_[b[:,2:4],b[:,4]]
    bounds=b/np.array([240.,240.,240.,240.,120.])
    tokens=np.zeros((len(x),64,40),np.float32)
    tokens[:,:16,:5]=bounds
    tokens[:,:16,5:17]=np.concatenate([(lo[None]-x[:,None,:3])/scale,(hi[None]-x[:,None,:3])/scale,
                                      (lo[None]-x[:,None,3:])/scale,(hi[None]-x[:,None,3:])/scale],-1)
    return np.c_[pose,tokens.reshape(len(x),-1)].astype(np.float32)
def features(name,variant):
    if variant=='RawMap':return np.load(EXP/f'cache/raw_map_{name}.npz')['features']
    if variant=='Coord':
        x=np.load(EXP/'cache/pool.npz')['unique'] if name=='pool' else np.load(EXP/f'cache/queries_{name}.npz')['queries']
        out=np.zeros((len(x),2566),np.float32);out[:,:6]=x/np.array([240.,240.,120.,240.,240.,120.]);return out
    return np.load(EXP/f'cache/features_{name}.npz')['G3']
def freeze_queries():
    dest=EXP/'cache/queries_confirm.npz'
    if dest.exists():return
    g=GeometryMap(config());p=np.load(EXP/'cache/pool.npz');train=np.unique(p['x'][p['train']],axis=0)
    tree=cKDTree(train[:,:3]);samples=[];drawn=fold_reject=buffer_reject=0
    # One seed and one draw stream, before any confirmation reference exists.
    q,_=g.sample_safe_queries(131072,2026102702)
    for x in q:
        drawn+=1
        if folds(x[None])[0]!=4:fold_reject+=1;continue
        near=tree.query_ball_point(x[:3],10.)
        if near and (g.road_distance(x[3:],train[near,3:])<=5.).any():buffer_reject+=1;continue
        samples.append(x)
        if len(samples)==4096:break
    assert len(samples)==4096
    np.savez_compressed(dest,queries=np.array(samples),weights=np.full(4096,1/4096),folds=np.full(4096,4))
    save(EXP/'configs/query_freeze_confirm.json',dict(n=4096,seed=2026102702,fold=4,sha256=sha(dest),drawn=drawn,fold_rejected=fold_reject,buffer_rejected=buffer_reject,reference_generated=False,created_at=stamp(),selection='safe air-volume x road-length conditioned on fold4 and outside joint training buffer (UAV10m AND road5m)'))
_geo=None
def init_geometry():
    global _geo
    from public_multibounce_geometry import PublicGeometry
    _geo=PublicGeometry(config())
def chunk(item):
    i,x=item;f,stats=_geo.features(x,level=2);return i,f,stats
def prepare(workers=24):
    freeze_queries();positions={'pool':np.load(EXP/'cache/pool.npz')['unique']}
    positions.update({n:np.load(EXP/f'cache/queries_{n}.npz')['queries'] for n in ['dev','confirm']})
    for name,x in positions.items():
        raw=EXP/f'cache/raw_map_{name}.npz'
        if not raw.exists():np.savez_compressed(raw,features=raw_map(x))
        dest=EXP/f'cache/features_{name}.npz'
        if dest.exists():continue
        f=np.empty((len(x),2566),np.float32);stats=[]
        with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn'),initializer=init_geometry) as pool:
            for i,a,s in pool.map(chunk,[(i,x[i:i+32]) for i in range(0,len(x),32)]):
                f[i:i+len(a)]=a;stats.extend(s)
        np.savez_compressed(dest,G3=f);save(EXP/f'results/geometry_{name}.json',stats)
    save(EXP/'results/data_ready.json',dict(created_at=stamp(),cache_hashes={p.name:sha(p) for p in sorted((EXP/'cache').glob('*.npz'))},reference_confirm_exists=(EXP/'cache/reference_confirm.npz').exists()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=24);prepare(p.parse_args().workers)
