"""Equations 14--16: committee disagreement and separate historical coverage."""
import numpy as np
from .model import committee_distances

def joint_neighborhood(candidates,positions,geometry,cfg):
    c,x=np.atleast_2d(candidates),np.atleast_2d(positions);a=cfg['acquisition']
    air=np.sum((c[:,None,:3]-x[None,:,:3])**2,-1)/a['history_uav_scale_m']**2
    road=geometry.road_distance(c[:,3:],x[:,3:])/a['history_ugv_road_scale_m']
    return np.exp(-(air+road**2)/2)

def committee_disagreement(predictions,cfg):return committee_distances(predictions,cfg)[1]

def measurement_value(disagreement,historical_coverage,cfg):
    a=cfg['acquisition'];e=a['exploration_epsilon']
    return (1-e)*disagreement/(max(disagreement)+a['normalization_delta'])+e*(1-historical_coverage)

def score_candidates(committee,geometry,history,cfg,seed,remaining_s):
    count=cfg['acquisition']['candidates']
    # Conservative rest-to-rest travel lower bound followed by return to fixed service site.
    speed=cfg['motion']['uav_speed_max_mps'];factor=1.875
    start=np.array(cfg['scene']['initial_uav_xyz_m'])
    # Uniform rejection in a reachable-volume bounding box and exact clipped road intervals.
    # This avoids an empty candidate pool in the short Smoke windows.
    radius=remaining_s*speed/(2*factor);gradius=remaining_s*cfg['motion']['ugv_speed_max_mps']/(2*factor)
    rng=np.random.Generator(np.random.PCG64DXSM(seed));intervals=[]
    for eid,(a,b) in enumerate(geometry.edges):
        length=geometry.lengths[eid];da=geometry.node_dist[0,a];db=geometry.node_dist[0,b]
        segments=[]
        if gradius>=da:segments.append((0,min(length,gradius-da)))
        if gradius>=db:segments.append((max(0,length-(gradius-db)),length))
        if len(segments)==2 and segments[0][1]>=segments[1][0]:segments=[(0,length)]
        for lo,hi in segments:
            if hi>lo:intervals.append((eid,lo,hi))
    if not intervals:raise ValueError('No reachable road interval')
    intervals=np.array(intervals);prob=(intervals[:,2]-intervals[:,1]);prob/=prob.sum()
    lo=np.maximum([0,0,geometry.s['uav_base_height_m']],start-radius)
    hi=np.minimum([*geometry.s['size_xy_m'],geometry.s['uav_max_height_m']],start+radius)
    chosen=[]
    for _ in range(100):
        air=rng.uniform(lo,hi,(max(64,4*count),3));ids=rng.choice(len(intervals),len(air),p=prob)
        eid=intervals[ids,0].astype(int);dist=rng.uniform(intervals[ids,1],intervals[ids,2])
        ground=geometry.nodes[geometry.edges[eid,0]]+(dist/geometry.lengths[eid])[:,None]*(geometry.nodes[geometry.edges[eid,1]]-geometry.nodes[geometry.edges[eid,0]])
        candidates=np.c_[air,ground,np.full(len(air),geometry.s['ugv_antenna_height_m'])]
        travel=factor*np.linalg.norm(air-start,axis=-1)*2/speed
        road=factor*geometry.road_distance(candidates[:,3:],[cfg['scene']['initial_ugv_xyz_m']])[:,0]*2/cfg['motion']['ugv_speed_max_mps']
        valid=geometry.air_safe(air)&(np.maximum(travel,road)<=remaining_s)
        chosen.extend(candidates[valid])
        if len(chosen)>=count:break
    if len(chosen)<count:raise ValueError('Reachable candidate sampling failed')
    c=np.array(chosen[:count]);w=np.full(count,1/count)
    travel=factor*(np.linalg.norm(c[:,:3]-start,axis=-1)+np.linalg.norm(c[:,:3]-start,axis=-1))/speed
    ground=factor*geometry.road_distance(c[:,3:],[cfg['scene']['initial_ugv_xyz_m']])[:,0]*2/cfg['motion']['ugv_speed_max_mps']
    reachable=np.maximum(travel,ground)<=remaining_s
    c,w=c[reachable],w[reachable]
    if not len(c):raise ValueError('No reachable candidates')
    w=w/w.sum();features=geometry.features(c);predictions=committee.predict_members(features)
    disagreement=committee_disagreement(predictions,cfg);coverage=np.zeros(len(c))
    for i in range(0,len(history),512):coverage=np.maximum(coverage,joint_neighborhood(c,history[i:i+512],geometry,cfg).max(-1))
    return c,w,measurement_value(disagreement,coverage,cfg),disagreement,coverage
