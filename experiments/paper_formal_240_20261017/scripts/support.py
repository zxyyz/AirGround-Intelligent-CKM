from common import *
import hashlib
import numpy as np
from ckm_sim.geometry import GeometryMap

OLD=ROOT/'experiments/disagreement_gain_new_physics_20261017'

def cfg_for(world):
    c=config();c.pop('active_branch',None);c['physics_world']=world
    c['runtime']['device']='cuda:0';c['runtime']['receiver_batch_size']=8
    c['propagation']['rx_zenith_interval_rad']=[0,float(np.pi)] if world=='W1a' else [0,float(np.pi/2)]
    c['propagation']['tx_zenith_interval_rad']=[0,float(np.pi)] if world=='W1a' else [float(np.pi/2),float(np.pi)]
    c['experiment']['policies']=['coverage_joint_v14','proposed_v14'];return c

def spatial_fold(x):
    bins=np.floor(np.asarray(x)[[0,1,2,3,4]]/np.array([60,60,30,60,60])).astype('<i8')
    return int.from_bytes(hashlib.blake2b(bins.tobytes(),digest_size=8,person=b'CKMLdev3').digest(),'little')%3

def cell_keys(x,g,air=20.,road=10.,k=None):
    x=np.asarray(x);edge,s=g.road_attach(x[:,3:]);a=np.floor(x[:,:3]/air).astype(int);b=np.floor(s/road).astype(int)
    keys=[tuple(map(int,[*aa,e,t])) for aa,e,t in zip(a,edge,b)]
    return keys if k is None else [(*key,int(kk)) for key,kk in zip(keys,k)]

def history_coverage(c,history,g,cfg):
    from ckm_sim.acquisition import joint_neighborhood
    h=np.zeros(len(c))
    for i in range(0,len(history),256):h=np.maximum(h,joint_neighborhood(c,history[i:i+256],g,cfg).max(-1))
    return h

def buffered(c,history,g):
    keep=np.ones(len(c),bool)
    for i in range(0,len(history),256):
        x=history[i:i+256]
        close=np.linalg.norm(c[:,None,:3]-x[None,:,:3],axis=-1)<=10
        keep &= ~(close & (g.road_distance(c[:,3:],x[:,3:])<=5)).any(-1)
    return keep

def csvread(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def safe_append(path,rows):
    rows=list(rows)
    if not rows:return
    path=Path(path)
    if path.exists():
        old=csvread(path);csvwrite(path,[*old,*rows])
    else:csvwrite(path,rows)

def unpack(padded,lengths):return [a[:k] for a,k in zip(padded,lengths)]

def pack(paths):
    out=np.zeros((len(paths),3,6));k=np.array([len(z) for z in paths])
    for i,z in enumerate(paths):out[i,:len(z)]=z
    return out,k
