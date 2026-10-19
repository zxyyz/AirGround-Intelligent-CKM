"""Current-state planning, fixed physical service deadline and shared controls."""
import copy
import numpy as np
from .planning import (initial_state,towards,Primitive,baseline_targets,recovery_steps,
    recovery_suffix,beam_plan)
from .acquisition import joint_neighborhood,committee_disagreement,measurement_value,score_candidates
from .v14_protocol import policy_kind

def route_prefix(actions,state,execute_s,cfg):
    n=round(execute_s/cfg['motion']['action_duration_s']);prefix=actions[:n]
    if len(prefix)!=n:raise ValueError('Planner returned too few actions')
    final=copy.deepcopy(state);final['x']=prefix[-1].end.copy()
    return prefix,final

def independent_segment(state,g,cfg,remaining_s,execute_s):
    """Preserve the original coverage cursors at the EXECUTED prefix endpoint."""
    state=copy.deepcopy(state);m=cfg['motion'];dt=m['action_duration_s'];n=round(remaining_s/dt)
    air,ground=baseline_targets(g);out=[];states=[]
    for step in range(n):
        x=state['x'];remain=n-step
        if recovery_steps(x,g,cfg)>=remain:break
        ac=state['air_cursor']%len(air);gc=state['ground_cursor']%len(ground)
        target=air[ac]
        a=towards(x[:3],target,m['uav_speed_max_mps']*dt/1.875,m['uav_vertical_speed_max_mps']*dt/1.875)
        gt=ground[gc];b=x[3:].copy()
        # Fix the road sequence by shortest axis-aligned attachment, never a diagonal.
        if abs(b[0]-gt[0])>1e-7 and min(abs(np.array(g.s['road_coordinates_m'])-b[1]))<1e-7:
            b[0]+=np.sign(gt[0]-b[0])*min(abs(gt[0]-b[0]),m['ugv_speed_max_mps']*dt/1.875)
        elif abs(b[1]-gt[1])>1e-7:
            b[1]+=np.sign(gt[1]-b[1])*min(abs(gt[1]-b[1]),m['ugv_speed_max_mps']*dt/1.875)
        action=Primitive(x,np.r_[a,b],dt,cfg)
        if not action.safe(g):
            # Public-geometry detour: climb vertically to the global roof envelope.
            cruise=max(x[2],float(g.buildings[:,4].max()+g.s['roof_clearance_m'])+.01)
            target=np.r_[x[:2],min(cruise,g.s['uav_max_height_m'])]
            a=towards(x[:3],target,m['uav_speed_max_mps']*dt/1.875,m['uav_vertical_speed_max_mps']*dt/1.875)
            action=Primitive(x,np.r_[a,b],dt,cfg)
            # Once at cruise, traverse horizontally before descending at the waypoint.
            if np.linalg.norm(a-x[:3])<1e-7:
                a=towards(x[:3],np.r_[air[ac,:2],x[2]],m['uav_speed_max_mps']*dt/1.875)
                action=Primitive(x,np.r_[a,b],dt,cfg)
        if not action.safe(g) or recovery_steps(action.end,g,cfg)>remain-1:break
        out.append(action);state['x']=action.end
        if np.linalg.norm(action.end[:3]-air[ac])<1e-7:state['air_cursor']+=1
        if np.linalg.norm(action.end[3:]-ground[gc])<1e-7:state['ground_cursor']+=1
        states.append(copy.deepcopy(state))
    suffix=recovery_suffix(state,g,cfg)
    for action in suffix:
        out.append(action);state['x']=action.end;states.append(copy.deepcopy(state))
    if len(out)>n:raise ValueError('Recovery exceeds physical deadline')
    while len(out)<n:
        out.append(Primitive(state['x'],state['x'],dt,cfg));states.append(copy.deepcopy(state))
    ne=round(execute_s/dt)
    return out[:ne],states[ne-1],{'policy':'independent','recovery_actions':len(suffix),'remaining_physical_s':remaining_s,'execute_s':execute_s}

def feasible_via_high_cruise(current,candidates,g,cfg,remaining_s):
    m=cfg['motion'];dt=m['action_duration_s'];dx=m['uav_speed_max_mps']*dt/1.875
    dz=m['uav_vertical_speed_max_mps']*dt/1.875;dg=m['ugv_speed_max_mps']*dt/1.875
    cruise=np.maximum(np.maximum(current[2],candidates[:,2]),g.buildings[:,4].max()+g.s['roof_clearance_m'])
    air=np.ceil((cruise-current[2])/dz-1e-9)+np.ceil(np.linalg.norm(candidates[:,:2]-current[:2],axis=-1)/dx-1e-9)+np.ceil((cruise-candidates[:,2])/dz-1e-9)
    # Manhattan road distance plus one action rounding per possible turn is conservative.
    road=np.ceil(g.road_distance(candidates[:,3:],[current[3:]])[:,0]/dg-1e-9)+2
    returning=np.array([recovery_steps(c,g,cfg) for c in candidates])
    return (np.maximum(air,road)+returning)*dt<=remaining_s

def nested_candidates(state,g,cfg,seed,remaining_s):
    count=cfg['acquisition']['candidates'];chosen=[]
    m=cfg['motion'];current=np.asarray(state['x']);anchor=initial_state(cfg)['x']
    radius=remaining_s*m['uav_speed_max_mps']/1.875
    vertical=remaining_s*m['uav_vertical_speed_max_mps']/1.875
    extent=np.array([radius,radius,vertical])
    low=np.maximum(np.maximum(current[:3]-extent,anchor[:3]-extent),[0,0,g.s['uav_base_height_m']])
    high=np.minimum(np.minimum(current[:3]+extent,anchor[:3]+extent),[*g.s['size_xy_m'],g.s['uav_max_height_m']])
    gradius=remaining_s*m['ugv_speed_max_mps']/1.875;intervals=[]
    for eid,(a,b) in enumerate(g.edges):
        length=g.lengths[eid];da=g.node_dist[0,a];db=g.node_dist[0,b];segments=[]
        if gradius>=da:segments.append((0,min(length,gradius-da)))
        if gradius>=db:segments.append((max(0,length-(gradius-db)),length))
        if len(segments)==2 and segments[0][1]>=segments[1][0]:segments=[(0,length)]
        intervals.extend((eid,lo,hi) for lo,hi in segments if hi>lo)
    if not intervals or np.any(high<=low):raise ValueError('No current-state candidate domain')
    intervals=np.asarray(intervals);prob=intervals[:,2]-intervals[:,1];prob/=prob.sum()
    rng=np.random.Generator(np.random.PCG64DXSM(seed))
    # Fixed-size batches independent of requested candidate count make nested pools.
    for block in range(16):
        air=rng.uniform(low,high,(4096,3));ids=rng.choice(len(intervals),4096,p=prob)
        eid=intervals[ids,0].astype(int);dist=rng.uniform(intervals[ids,1],intervals[ids,2])
        road=g.nodes[g.edges[eid,0]]+(dist/g.lengths[eid])[:,None]*(g.nodes[g.edges[eid,1]]-g.nodes[g.edges[eid,0]])
        x=np.c_[air,road,np.full(len(air),g.s['ugv_antenna_height_m'])]
        valid=g.air_safe(air)&feasible_via_high_cruise(state['x'],x,g,cfg,remaining_s);chosen.extend(x[valid])
        if len(chosen)>=count:break
    if len(chosen)<count:raise ValueError('Insufficient feasible current-state candidate pool')
    c=np.asarray(chosen[:count]);return c,np.full(count,1/count)

def joint_scores(policy,committee,g,history,state,cfg,seed,remaining_s):
    policy_kind(policy)
    if policy=='proposed':return score_candidates(committee,g,history,cfg,seed,remaining_s)
    legacy=(cfg['acquisition'].get('candidate_domain')=='legacy_when_anchor' and
            np.allclose(state['x'],initial_state(cfg)['x'],rtol=0,atol=1e-8))
    if legacy:
        c,w,j,u,coverage=score_candidates(committee,g,history,cfg,seed,remaining_s)
        if policy=='coverage_joint_v14':j=1-coverage
        elif policy=='disagreement_only_v14':j=u/(max(u)+cfg['acquisition']['normalization_delta'])
        return c,w,j,u,coverage
    c,w=nested_candidates(state,g,cfg,seed,remaining_s);coverage=np.zeros(len(c))
    for i in range(0,len(history),512):coverage=np.maximum(coverage,joint_neighborhood(c,history[i:i+512],g,cfg).max(-1))
    u=np.zeros(len(c)) if policy=='coverage_joint_v14' else committee_disagreement(committee.predict_members(g.features(c)),cfg)
    if policy=='coverage_joint_v14':j=1-coverage
    elif policy=='disagreement_only_v14':j=u/(max(u)+cfg['acquisition']['normalization_delta'])
    else:j=measurement_value(u,coverage,cfg)
    return c,w,j,u,coverage

def plan_segment(policy,committee,g,history,state,cfg,seed,remaining_s,execute_s):
    if policy_kind(policy)=='independent':return (*independent_segment(state,g,cfg,remaining_s,execute_s),None)
    c,w,j,u,h=joint_scores(policy,committee,g,history,state,cfg,seed,remaining_s)
    actions,final,log=beam_plan(state,g,cfg,remaining_s,c,w,j)
    prefix,final=route_prefix(actions,state,execute_s,cfg)
    log.update(policy=policy,remaining_physical_s=remaining_s,execute_s=execute_s,
        action_space='UAV axial/vertical/hover x UGV legal road/hover, rest-to-rest quintic')
    return prefix,final,log,{'positions':c,'weights':w,'score':j,'disagreement':u,'historical_coverage':h,
        'start_state':np.asarray(state['x']),'remaining_physical_s':remaining_s}
