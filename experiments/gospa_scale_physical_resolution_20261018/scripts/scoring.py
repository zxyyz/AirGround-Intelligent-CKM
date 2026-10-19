"""Independent float64, vectorized exact partial assignment for K<=3.

No dependency on legacy metrics: that implementation is a test oracle only.
All four physical error terms are recomputed from fixed saved path sets.
"""
from common import *
from itertools import combinations,permutations

def directions(a):
    phi,theta=a[...,0],a[...,1]
    return np.stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)],-1)

def components(pred,ref):
    a=np.asarray(pred)[:,:,None,:];b=np.asarray(ref,dtype=np.float64)[:,None,:,:]
    terms=[(a[...,0]-b[...,0])**2,(a[...,1]-b[...,1])**2]
    for j in [2,4]:
        u,v=directions(a[...,j:j+2]),directions(b[...,j:j+2])
        angle=np.arctan2(np.linalg.norm(np.cross(u,v),axis=-1),np.sum(u*v,axis=-1))
        terms.append(angle**2)
    return np.stack(terms,-1)

def assignments(n,m):
    yield ()
    for k in range(1,min(n,m)+1):
        for a in combinations(range(n),k):
            for b in permutations(range(m),k):yield tuple(zip(a,b))

def match(terms,kpred,kref,scale):
    sg,st,sa=scale['gain_db'],scale['delay_ns']*1e-9,np.deg2rad(scale['angle_deg'])
    cost=terms[...,0]/sg**2+terms[...,1]/st**2+terms[...,2]/sa**2+terms[...,3]/sa**2
    cutoff=scale['cutoff']**2;n=len(kpred)
    # Code bit (i*3+j) gives an assignment-set signature, invariant to enumeration.
    codes=np.zeros(n,np.uint16);best=cutoff*(kpred+kref)/2.;nmatch=np.zeros(n,np.int8);param=np.zeros(n)
    for kp in range(4):
        for kr in range(4):
            ix=np.flatnonzero((kpred==kp)&(kref==kr))
            if not len(ix):continue
            current=best[ix].copy();local=np.zeros(len(ix),np.uint16);nm=np.zeros(len(ix),np.int8);pc=np.zeros(len(ix))
            for pairs in assignments(kp,kr):
                if not pairs:continue
                c=sum(cost[ix,i,j] for i,j in pairs)
                value=c+cutoff*(kp+kr-2*len(pairs))/2
                keep=value<current-1e-12
                current[keep]=value[keep];local[keep]=sum(1<<(i*3+j) for i,j in pairs);nm[keep]=len(pairs);pc[keep]=c[keep]
            best[ix]=current;codes[ix]=local;nmatch[ix]=nm;param[ix]=pc
    miss=kref-nmatch;false=kpred-nmatch
    np.testing.assert_allclose(best,param+cutoff/2*(miss+false),rtol=1e-13,atol=1e-12)
    return dict(squared_gospa=best,parameter_cost=param,miss_cost=miss*cutoff/2,false_cost=false*cutoff/2,nmatch=nmatch,nmiss=miss,nfalse=false,assignment_code=codes)

def score(pred,kpred,ref,kref,second,weights,scale,terms=None,baseline_codes=None):
    # Legacy outputs were float32 before being copied into float64 storage.
    # Preserve its trig evaluation precision to reproduce S0 pointwise.
    pred=np.asarray(pred,dtype=np.float32);ref=np.asarray(ref,dtype=np.float64)
    d=match(components(pred,ref) if terms is None else terms,kpred,kref,scale)
    n=len(kpred);sums=np.zeros((n,6));counts=np.zeros((n,6),int);second_match=np.zeros(n,int)
    tol=1e-3
    for i in range(3):
        for j in range(3):
            keep=(d['assignment_code']&(1<<(i*3+j)))!=0
            if not keep.any():continue
            a,b=pred[keep,i],ref[keep,j];delta=a-b;delta[:,[2,4]]=(delta[:,[2,4]]+np.pi)%(2*np.pi)-np.pi
            valid=np.ones(delta.shape,bool)
            for az in [2,4]:valid[:,az]=np.minimum(abs(np.sin(a[:,az+1])),abs(np.sin(b[:,az+1])))>tol
            sums[keep]+=np.where(valid,delta**2,0);counts[keep]+=valid;second_match[keep]+=second[keep,j]
    groups={'all':np.ones(n,bool),'empty':kref==0,'nonempty':kref>0,'K1':kref==1,'K2plus':kref>=2,'second_related':second.any(-1)};rows=[]
    names=['gain_db','delay_ns','aoa_az_deg','aoa_zen_deg','aod_az_deg','aod_zen_deg']
    for name,keep in groups.items():
        ix=np.flatnonzero(keep);mass=float(weights[ix].sum());row=dict(subset=name,n_queries=len(ix),population_mass=mass)
        if not len(ix):row.update(global_gospa=np.nan,second_component_total=0,second_matched_recall=np.nan);rows.append(row);continue
        w=weights[ix]/mass;matched=int(d['nmatch'][ix].sum());nr=int(kref[ix].sum());npred=int(kpred[ix].sum());n2=int(second[ix].sum())
        rms=np.sqrt(np.divide((w[:,None]*sums[ix]).sum(0),(w[:,None]*counts[ix]).sum(0),out=np.full(6,np.nan),where=(w[:,None]*counts[ix]).sum(0)>0));rms[1]*=1e9;rms[2:]*=180/np.pi
        row.update(global_gospa=float(np.sqrt(w@d['squared_gospa'][ix])),mean_squared_gospa=float(w@d['squared_gospa'][ix]),gospa_p90=float(np.sqrt(np.quantile(d['squared_gospa'][ix],.9))),matched_parameter_cost=float(w@d['parameter_cost'][ix]),miss_cost=float(w@d['miss_cost'][ix]),false_cost=float(w@d['false_cost'][ix]),n_matched=matched,n_missed=int(d['nmiss'][ix].sum()),n_false=int(d['nfalse'][ix].sum()),mean_miss=float(w@d['nmiss'][ix]),mean_false=float(w@d['nfalse'][ix]),reference_components=nr,predicted_components=npred,matched_recall=matched/nr if nr else np.nan,precision=matched/npred if npred else np.nan,second_component_total=n2,second_component_matched=int(second_match[ix].sum()),second_matched_recall=float(second_match[ix].sum()/n2) if n2 else np.nan,count_accuracy=float(w@(kpred[ix]==kref[ix])),**{'rmse_'+k:float(v) for k,v in zip(names,rms)},**{'n_'+k:int(v) for k,v in zip(names,counts[ix].sum(0))})
        if baseline_codes is not None:
            changed=d['assignment_code'][ix]!=baseline_codes[ix];active=(kpred[ix]>0)&(kref[ix]>0)
            row.update(assignment_changed_n=int(changed.sum()),assignment_changed_fraction=float(w@changed),assignment_changed_active_fraction=float(changed[active].mean()) if active.any() else np.nan)
        rows.append(row)
    d['second_matched']=second_match
    return rows,d
