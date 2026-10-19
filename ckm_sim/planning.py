"""Safe rest-to-rest quintic primitives, exact sampled coverage, beam search.

The UAV is a multirotor and may stop/hover. Direction changes occur at zero
velocity/acceleration; all moving pieces are straight, with curvature zero.
The historical coverage route permits multi-axis UAV moves; the historical
beam search uses axial/vertical/hover candidates. Both share primitive physics
and the same airborne service suffix. v1.4 joint controls share one beam planner.
"""
import copy
import itertools
import numpy as np
from .acquisition import joint_neighborhood

def smoothstep(t):return t**3*(10+t*(-15+6*t))
def smoothstep_derivative(t):return 30*t**2*(1-t)**2

class Primitive:
    def __init__(self,start,end,duration,cfg):
        self.start=np.asarray(start,float);self.end=np.asarray(end,float);self.duration=float(duration)
        self.delta=self.end-self.start;self.cfg=cfg
        m=cfg['motion'];factor=1.875/self.duration
        if np.linalg.norm(self.delta[:3])*factor>m['uav_speed_max_mps']+1e-8:raise ValueError('UAV speed')
        if abs(self.delta[2])*factor>m['uav_vertical_speed_max_mps']+1e-8:raise ValueError('Vertical speed')
        if np.linalg.norm(self.delta[3:])*factor>m['ugv_speed_max_mps']+1e-8:raise ValueError('UGV speed')

    def position(self,t):return self.start+smoothstep(np.asarray(t)/self.duration)[...,None]*self.delta
    def velocity(self,t):return smoothstep_derivative(np.asarray(t)/self.duration)[...,None]*self.delta/self.duration
    def safe(self,g):
        step=self.cfg['motion']['integration_step_s']
        t=np.linspace(0,self.duration,int(np.ceil(self.duration/step))+1);x=self.position(t)
        if not g.air_safe(x[:,:3]).all():return False
        # A conservative XY box encloses every point on the straight primitive.
        b=g.buildings;low=np.minimum(self.start[:2],self.end[:2]);high=np.maximum(self.start[:2],self.end[:2])
        d=np.maximum(np.maximum(b[:,:2]-high,low-b[:,2:4]),0)
        near=np.linalg.norm(d,axis=-1)<=g.s['horizontal_clearance_m']+1e-8
        floor=max(g.s['uav_base_height_m'],np.max(np.where(near,b[:,4]+g.s['roof_clearance_m'],0)))
        if min(self.start[2],self.end[2])<floor-1e-8:return False
        try:g.validate_joint_state(x)
        except ValueError:return False
        # Ground action must remain on one road; no diagonal shortcut.
        return abs(self.delta[3])<1e-8 or abs(self.delta[4])<1e-8

    def serialize(self):return {'start':self.start.tolist(),'end':self.end.tolist(),'duration_s':self.duration}

def initial_state(cfg):
    return {'x':np.r_[cfg['scene']['initial_uav_xyz_m'],cfg['scene']['initial_ugv_xyz_m']],
            'air_cursor':0,'ground_cursor':0}

def towards(p,target,maximum,vertical_max=None):
    delta=np.asarray(target)-p;norm=np.linalg.norm(delta)
    fraction=min(1,maximum/max(norm,1e-30))
    if vertical_max is not None:fraction=min(fraction,vertical_max/max(abs(delta[2]),1e-30))
    return p+delta*fraction

def ground_options(p,g,cfg):
    length=cfg['motion']['ugv_speed_max_mps']*cfg['motion']['action_duration_s']/1.875
    roads=np.array(g.s['road_coordinates_m']);out=[p.copy()]
    for axis in [0,1]:
        if np.min(abs(roads-p[1-axis]))>1e-7:continue
        for sign in [-1,1]:
            ahead=roads[sign*(roads-p[axis])>1e-7]
            if not len(ahead):continue
            target=np.min(ahead) if sign==1 else np.max(ahead)
            q=p.copy();q[axis]+=sign*min(length,abs(target-p[axis]));out.append(q)
    return out

def air_options(p,g,cfg):
    m=cfg['motion'];distance=m['uav_speed_max_mps']*m['action_duration_s']/1.875
    dz=m['uav_vertical_speed_max_mps']*m['action_duration_s']/1.875
    vectors=[[0,0,0],[distance,0,0],[-distance,0,0],[0,distance,0],[0,-distance,0],[0,0,dz],[0,0,-dz]]
    return [p+v for v in np.array(vectors)]

def recovery_suffix(state,g,cfg):
    """Explicit feasible paths to the same legal road site for BOTH strategies."""
    m=cfg['motion'];duration=m['action_duration_s'];dx=m['uav_speed_max_mps']*duration/1.875
    dz=m['uav_vertical_speed_max_mps']*duration/1.875;dg=m['ugv_speed_max_mps']*duration/1.875
    x=np.array(state['x']);anchor=np.r_[cfg['scene']['initial_uav_xyz_m'],cfg['scene']['initial_ugv_xyz_m']]
    cruise=max(float(g.buildings[:,4].max()+g.s['roof_clearance_m']),anchor[2],x[2])
    atargets=[np.r_[x[:2],cruise],np.r_[anchor[:2],cruise],anchor[:3]]
    roads=np.array(g.s['road_coordinates_m'])
    if np.min(abs(roads-x[4]))<1e-7:gtargets=[np.array([anchor[3],x[4],x[5]]),anchor[3:]]
    else:gtargets=[np.array([x[3],anchor[4],x[5]]),anchor[3:]]
    ai=gi=0;out=[]
    for _ in range(300):
        while ai<len(atargets) and np.linalg.norm(x[:3]-atargets[ai])<1e-7:ai+=1
        while gi<len(gtargets) and np.linalg.norm(x[3:]-gtargets[gi])<1e-7:gi+=1
        if ai==len(atargets) and gi==len(gtargets):return out
        a=x[:3] if ai==len(atargets) else towards(x[:3],atargets[ai],dx,dz)
        b=x[3:] if gi==len(gtargets) else towards(x[3:],gtargets[gi],dg)
        action=Primitive(x,np.r_[a,b],duration,cfg)
        if not action.safe(g):raise ValueError('Unsafe recovery suffix')
        out.append(action);x=action.end
    raise RuntimeError('Recovery did not terminate')

def recovery_steps(x,g,cfg):
    """Same conservative count as recovery_suffix, without constructing primitives."""
    m=cfg['motion'];duration=m['action_duration_s']
    dx=m['uav_speed_max_mps']*duration/1.875;dz=m['uav_vertical_speed_max_mps']*duration/1.875
    dg=m['ugv_speed_max_mps']*duration/1.875;u=np.array(cfg['scene']['initial_uav_xyz_m'])
    cruise=max(float(g.buildings[:,4].max()+g.s['roof_clearance_m']),u[2],x[2])
    air=int(np.ceil(max(0,cruise-x[2])/dz-1e-9)+np.ceil(np.linalg.norm(x[:2]-u[:2])/dx-1e-9)+np.ceil(max(0,cruise-u[2])/dz-1e-9))
    ground=int(np.ceil(abs(x[3])/dg-1e-9)+np.ceil(abs(x[4])/dg-1e-9))
    return max(air,ground)

def trajectory_utility(c,w,j,actions,g,cfg):
    cover=np.zeros(len(c));offset=0.;f=cfg['experiment']['probe_frequency_hz']
    for action in actions:
        t=(np.arange(round(action.duration*f))+.5)/f
        cover=np.maximum(cover,joint_neighborhood(c,action.position(t),g,cfg).max(-1));offset+=action.duration
    return float(np.dot(w*j,cover)),cover

def baseline_targets(g):
    air=[]
    for layer in range(3):
        for row,y in enumerate([20,60,100,140,180,220]):
            xs=[20,60,100,140,180,220]
            if row%2:xs=xs[::-1]
            for x in xs:
                floor=float(g.floor([x,y]));height=[floor+1,(floor+g.s['uav_max_height_m'])/2,g.s['uav_max_height_m']-1][layer]
                air.append([x,y,height])
    ground=[];roads=g.s['road_coordinates_m']
    for row,y in enumerate(roads):
        for x in (roads if row%2==0 else roads[::-1]):ground.append([x,y,g.s['ugv_antenna_height_m']])
    # The transposed sweep traverses the perpendicular road family.
    for col,x in enumerate(roads):
        for y in (roads[::-1] if col%2==0 else roads):ground.append([x,y,g.s['ugv_antenna_height_m']])
    return np.array(air),np.array(ground)

def independent_plan(state,g,cfg,duration):
    state=copy.deepcopy(state);m=cfg['motion'];dt=m['action_duration_s'];n=round(duration/dt)
    air,ground=baseline_targets(g);out=[]
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
    suffix=recovery_suffix(state,g,cfg);out+=suffix
    if len(out)>n:raise ValueError('Recovery exceeds budget')
    x=out[-1].end if out else state['x']
    out += [Primitive(x,x,dt,cfg) for _ in range(n-len(out))];state['x']=out[-1].end
    return out,state,{'policy':'independent','recovery_actions':len(suffix),'planned_actions':len(out)}

def beam_plan(state,g,cfg,duration,c,w,j):
    dt=cfg['motion']['action_duration_s'];n=round(duration/dt);f=cfg['experiment']['probe_frequency_hz']
    probe_t=(np.arange(round(dt*f))+.5)/f
    beam=[(0.,copy.deepcopy(state),np.zeros(len(c)),[])]
    rejected=0
    for step in range(n):
        expanded=[];remaining=n-step
        for score,current,coverage,route in beam:
            x=current['x']
            if recovery_steps(x,g,cfg)>=remaining:
                expanded.append((score,current,coverage,route));continue
            for a,b in itertools.product(air_options(x[:3],g,cfg),ground_options(x[3:],g,cfg)):
                end=np.r_[a,b]
                if recovery_steps(end,g,cfg)>remaining-1:rejected+=1;continue
                try:action=Primitive(x,end,dt,cfg)
                except ValueError:rejected+=1;continue
                if not action.safe(g):rejected+=1;continue
                increment=joint_neighborhood(c,action.position(probe_t),g,cfg).max(-1)
                new=np.maximum(coverage,increment);value=float(np.dot(w*j,new))
                newstate=copy.deepcopy(current);newstate['x']=end
                expanded.append((value,newstate,new,route+[action]))
        if not expanded:break
        expanded.sort(key=lambda x:x[0],reverse=True)
        # Keep distinct end states/coverage, avoiding an all-hover duplicated beam.
        beam=[];seen=set()
        for item in expanded:
            key=(tuple(np.round(item[1]['x'],6)),tuple(np.round(item[2],5)))
            if key in seen:continue
            seen.add(key);beam.append(item)
            if len(beam)>=cfg['motion']['beam_width']:break
        if all(recovery_steps(item[1]['x'],g,cfg)>=n-len(item[3]) for item in beam):break
    complete=[]
    for _,current,coverage,route in beam:
        suffix=recovery_suffix(current,g,cfg)
        if len(route)+len(suffix)>n:continue
        actions=route+suffix;x=actions[-1].end if actions else current['x']
        actions += [Primitive(x,x,dt,cfg) for _ in range(n-len(actions))]
        score,coverage=trajectory_utility(c,w,j,actions,g,cfg)
        final=copy.deepcopy(current);final['x']=actions[-1].end
        complete.append((score,actions,final,len(suffix)))
    if not complete:
        actions,final,log=independent_plan(state,g,cfg,duration)
        log.update(policy='proposed',failure_reason='no_complete_beam_safe_baseline_suffix')
        return actions,final,log
    score,actions,final,recovery=max(complete,key=lambda item:item[0])
    return actions,final,{'policy':'proposed','utility':score,'rejected_actions':rejected,'recovery_actions':recovery,'beam_width':cfg['motion']['beam_width']}

def sample_trajectory(actions,cfg):
    f=cfg['experiment']['probe_frequency_hz'];t=[];x=[];velocity=[];offset=0
    for a in actions:
        local=(np.arange(round(a.duration*f))+.5)/f
        t.extend(offset+local);x.extend(a.position(local));velocity.extend(a.velocity(local));offset+=a.duration
    return np.array(t),np.array(x),np.array(velocity)

def budget_ledger(cfg):
    e=cfg['experiment'];return [e['initial_sensing_time_s']+r*e['round_time_s'] for r in range(e['rounds_after_initial']+1)]
