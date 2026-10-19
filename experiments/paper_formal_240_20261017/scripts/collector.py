"""Receiver-only observation pipeline. No model or evaluation input."""
from common import *
import numpy as np,torch,h5py,fcntl
from physics import SignalWorld,frontend,fold
from ckm_sim.sounding import simulate_received
from ckm_sim.planning import sample_trajectory
from store import put_window

_world=None
_cfg=None
_namespace=None
def worker_init(cfg):
    global _world,_cfg,_namespace
    _cfg=cfg;torch.set_num_threads(2);_world=SignalWorld('W1a',cfg)
    _namespace=hashlib.sha256((sha(EXP/'configs/effective_config.json')+sha(EXP/'configs/mesh_M2.json')+torch.cuda.get_device_name(0)).encode()).hexdigest()[:24]

def observe(request):
    positions,pair,r,indices=request;signals=[];counts=[];start=time.perf_counter()
    # Exact, lossless simulator cache only. The learner never receives channels
    # or RT counts; fresh noise and the full RF estimator run at every attempt.
    keys=[_namespace+'/'+hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest() for x in positions];cached={}
    with (EXP/'cache/rt_channels.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with h5py.File(EXP/'cache/rt_channels.h5','a') as cache:
            for key in keys:
                if key in cache:cached[key]=(cache[key][:],cache[key].attrs['counts'])
        fcntl.flock(lock,fcntl.LOCK_UN)
    added={}
    for x,i,key in zip(positions,indices,keys):
        if key in cached:h,c=cached[key]
        elif key in added:h,c=added[key]
        else:
            h,rec,_,_=_world.channel(x);c=np.array([len(rec['paths']),rec['signal_physical_k'],rec['target_physical_k']],dtype=np.int32);added[key]=(h,c)
        s,_=simulate_received(h,(pair,r,int(i)),_cfg);signals.append(s);counts.append(c)
    if added:
        with (EXP/'cache/rt_channels.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            with h5py.File(EXP/'cache/rt_channels.h5','a') as cache:
                for key,(h,c) in added.items():
                    if key not in cache:
                        d=cache.create_dataset(key,data=h,compression='gzip',compression_opts=1,shuffle=True);d.attrs['counts']=c
            fcntl.flock(lock,fcntl.LOCK_UN)
    return np.stack(signals),np.array(counts,dtype=np.int32),time.perf_counter()-start

class Receiver:
    def __init__(self,cfg):self.estimator,self.cal=frontend(cfg)
    def estimate(self,signals,pilots):
        zz=[];qq=[]
        for i in range(0,len(signals),8):
            try:z,q=self.estimator.estimate_many(signals[i:i+8],pilots)
            except Exception as exc:
                print('RF_BATCH_FALLBACK',repr(exc),flush=True);z=[];q=[]
                for s in signals[i:i+8]:
                    try:a,b=self.estimator.estimate(s,pilots)
                    except Exception as err:a=np.empty((0,6));b=dict(valid=False,reason=repr(err))
                    z.append(a);q.append(b)
            zz.extend(fold(a) for a in z);qq.extend(q)
        return zz,qq

def collect(phase,pair,stream,r,actions,state,plan,planning_s,g,receiver,pool,cfg):
    start=time.perf_counter();t,x,velocity=sample_trajectory(actions,cfg);g.validate_joint_state(x)
    assert len(t)==1800;assert np.allclose(state['x'],np.r_[cfg['scene']['initial_uav_xyz_m'],cfg['scene']['initial_ugv_xyz_m']])
    jobs=[(i,pool.submit(observe,(x[i:i+32],pair,r,np.arange(i,min(i+32,len(x)))))) for i in range(0,len(x),32)]
    z=np.zeros((len(x),3,6));k=np.zeros(len(x),np.int32);valid=np.zeros(len(x),bool);quality=[];raw=[];raw_ix=[];physical=[];rt_s=rf_s=0.
    pilots=np.eye(8)*np.sqrt(cfg['sounding']['pilot_energy_ep'])
    for first,future in jobs:
        signals,counts,cost=future.result();rt_s+=cost;physical.extend(counts)
        before=time.perf_counter();zs,qs=receiver.estimate(signals,pilots);rf_s+=time.perf_counter()-before
        for j,(a,q) in enumerate(zip(zs,qs)):
            i=first+j;k[i]=len(a);z[i,:len(a)]=a;valid[i]=bool(q['valid']);quality.append(json.dumps(q,default=lambda a:a.tolist()))
            global_i=r*1800+i
            if phase=='smoke' or i<8 or global_i%100==0 or not valid[i]:raw.append(signals[j]);raw_ix.append(i)
        if first%320==0 or first+32>=len(x):print('COLLECT',phase,pair,stream,r,min(first+32,len(x)),flush=True)
    landing=np.array(state['x']).copy();landing[2]=cfg['scene']['ugv_antenna_height_m']
    assert g.clear(np.array(state['x'])[:3],landing[:3])
    leg=1.875*abs(state['x'][2]-landing[2])/cfg['motion']['uav_vertical_speed_max_mps'];service=2*leg+60
    assert abs(service-427.96875)<1e-9
    arrays=dict(x=x,velocity=velocity,labels=z,n_paths=k,valid=valid,quality=np.array(quality),probe_index=np.arange(1800),global_probe_index=r*1800+np.arange(1800),time_s=r*1800+t,physical_time_s=r*(1800+service)+t,raw_index=np.array(raw_ix,np.int32),S=np.array(raw),physical_counts=np.array(physical))
    timing=dict(ray_trace_s=rt_s,rf_estimation_s=rf_s,collection_wall_s=time.perf_counter()-start,planning_s=planning_s)
    meta=dict(phase=phase,pair=pair,stream='shared' if r==0 else stream,round=r,state=state,plan=plan,actions=[a.serialize() for a in actions],service=dict(descent_s=leg,dwell_s=60,ascent_s=leg,total_s=service,sensing_probes=0),timing=timing,noise_key='(pair,round,local_probe_index)',label_source='noisy S -> frozen MUSIC+LS',retention='all for engineering smoke; first8 per window plus global_index%100==0 plus invalid for formal')
    return put_window(phase,pair,stream,r,arrays,meta),meta
