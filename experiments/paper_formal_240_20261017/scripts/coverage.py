"""Original beam semantics with a scalar trajectory-coverage objective callback.
Source copied from ckm_sim/planning.py; only objective and exact kernel memoization vary.
"""
from common import *
import itertools,numpy as np
from ckm_sim.planning import *
from ckm_sim.acquisition import joint_neighborhood,committee_disagreement
from ckm_sim.v14_planning import nested_candidates
from support import history_coverage

def objective_beam(state,g,cfg,duration,c,objective,kernel_cache=None):
    if kernel_cache is None:kernel_cache={}
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
                key=tuple(np.r_[action.start,action.end])
                if key not in kernel_cache:kernel_cache[key]=joint_neighborhood(c,action.position(probe_t),g,cfg).max(-1)
                increment=kernel_cache[key]
                new=np.maximum(coverage,increment);value=float(objective(new))
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
        coverage=np.zeros(len(c))
        for action in actions:
            key=tuple(np.r_[action.start,action.end])
            if key not in kernel_cache:kernel_cache[key]=joint_neighborhood(c,action.position(probe_t),g,cfg).max(-1)
            coverage=np.maximum(coverage,kernel_cache[key])
        score=float(objective(coverage))
        final=copy.deepcopy(current);final['x']=actions[-1].end
        complete.append((score,actions,final,len(suffix)))
    if not complete:
        actions,final,log=independent_plan(state,g,cfg,duration)
        log.update(policy='proposed',failure_reason='no_complete_beam_safe_baseline_suffix')
        return actions,final,log
    score,actions,final,recovery=max(complete,key=lambda item:item[0])
    return actions,final,{'policy':'proposed','utility':score,'rejected_actions':rejected,'recovery_actions':recovery,'beam_width':cfg['motion']['beam_width']}

def fdelta(h,k):return float(np.maximum(k-h,0).mean())
def marginal_objective(nc,w,u,hq,scales):
    u=u/(max(u)+1e-8)
    return lambda k:.5*float(np.dot(w*u,k[:nc]))/scales['disagreement']+.5*fdelta(hq,k[nc:])/scales['coverage']
_PLAN_KERNELS={}
def plan(policy,committee,g,history,state,cfg,seed,selection):
    c,w=nested_candidates(state,g,cfg,seed,1800);h=history_coverage(c,history,g,cfg)
    u=np.zeros(len(c)) if policy=='CG' else committee_disagreement(committee.predict_members(g.features(c)),cfg)
    un=u/(max(u)+cfg['acquisition']['normalization_delta']);mode=selection['P1_mode'] if policy=='P1' else policy
    if mode=='B2':
        q=np.load(EXP/'cache/queries_public.npz')['queries'];hq=history_coverage(q,history,g,cfg)
        points=np.r_[c,q];objective=marginal_objective(len(c),w,u,hq,selection['scales']);score=np.full(len(c),np.nan)
    else:
        eps=1. if policy=='CG' else .1 if policy=='P0' else .5
        score=(1-eps)*un+eps*(1-h);points=c;objective=lambda k:float(np.dot(w*score,k))
    # Public geometry kernels are identical for the same candidate coordinates
    # across policy/learner branches. Cache exact arrays, without score reuse.
    cachekey=hashlib.sha256(points.tobytes()).hexdigest()
    cache=_PLAN_KERNELS.setdefault(cachekey,{})
    actions,final,log=objective_beam(state,g,cfg,1800,points,objective,cache)
    log.update(policy=policy,mode=mode,epsilon=None if mode=='B2' else eps)
    return actions,final,log,dict(positions=c,weights=w,score=score,disagreement=u,historical_coverage=h,start_state=state['x'])
