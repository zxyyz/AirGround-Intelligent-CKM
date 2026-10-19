"""Equations 7--9,19. Exact partial assignment, tie prefers unmatched."""
from itertools import combinations, permutations
import numpy as np
from .geometry import directions

def spherical_angle(u,v):
    return np.arctan2(np.linalg.norm(np.cross(u,v),axis=-1),np.sum(u*v,axis=-1))

def path_squared_distance(a,b,cfg):
    a,b=np.asarray(a),np.asarray(b);m=cfg['metric']
    return ((a[...,0]-b[...,0])/m['gain_scale_db'])**2+((a[...,1]-b[...,1])/m['delay_scale_s'])**2+sum(
        (spherical_angle(directions(a[...,j:j+2]),directions(b[...,j:j+2]))/
         np.deg2rad(m[key]))**2 for j,key in [(2,'aoa_direction_scale_deg'),(4,'aod_direction_scale_deg')])

def partial_assignments(n,m):
    yield ()
    for k in range(1,min(n,m)+1):
        for a in combinations(range(n),k):
            for b in permutations(range(m),k):yield tuple(zip(a,b))

def squared_gospa(a,b,cfg):
    a,b=np.asarray(a).reshape(-1,6),np.asarray(b).reshape(-1,6)
    cutoff=cfg['metric']['gospa_c']**2; best=cutoff*(len(a)+len(b))/2;assignment=()
    cost=path_squared_distance(a[:,None],b[None,:],cfg)
    for pairs in partial_assignments(len(a),len(b)):
        value=sum(cost[i,j] for i,j in pairs)+cutoff*(len(a)+len(b)-2*len(pairs))/2
        if value<best-1e-12:best=float(value);assignment=pairs
    return best,assignment,len(b)-len(assignment),len(a)-len(assignment)

def global_gospa(prediction,reference,weights,cfg):
    return float(np.sqrt(np.dot(weights,[squared_gospa(a,b,cfg)[0] for a,b in zip(prediction,reference)])))

def parameter_rmse(prediction,reference,weights,cfg):
    sums=np.zeros(6);counts=np.zeros(6,int);weighted_counts=np.zeros(6);miss=false=matched=0
    errors=[];tol=cfg['metric']['azimuth_pole_sin_tolerance']
    for a,b,w in zip(prediction,reference,weights):
        d,pairs,nmiss,nfalse=squared_gospa(a,b,cfg);errors.append(d)
        miss+=w*nmiss;false+=w*nfalse;matched+=len(pairs)
        for i,j in pairs:
            delta=a[i]-b[j];delta[[2,4]]=(delta[[2,4]]+np.pi)%(2*np.pi)-np.pi
            valid=np.ones(6,bool)
            for az in [2,4]:valid[az]=min(abs(np.sin(a[i,az+1])),abs(np.sin(b[j,az+1])))>tol
            sums+=w*np.where(valid,delta**2,0);counts+=valid;weighted_counts+=w*valid
    rms=np.sqrt(np.divide(sums,weighted_counts,out=np.full(6,np.nan),where=weighted_counts>0))
    rms[1]*=1e9;rms[2:]*=180/np.pi
    names=['gain_db','delay_ns','aoa_az_deg','aoa_zen_deg','aod_az_deg','aod_zen_deg']
    result={'global_gospa':float(np.sqrt(np.dot(weights,errors))),
            'gospa_p90':float(np.sqrt(np.quantile(errors,.9))), 'mean_miss':float(miss),'mean_false':float(false),'n_matched':matched}
    result.update({'rmse_'+k:float(v) for k,v in zip(names,rms)})
    result.update({'n_'+k:int(v) for k,v in zip(names,counts)})
    return result,np.array(errors)
