"""Isolated evaluation process. Metrics never enter the trajectory workers."""
from common import *
import argparse
import numpy as np,torch
from formal_models import make_model
from decoding import raw_predict,decode
from ckm_sim.metrics import squared_gospa,path_squared_distance

def score(pred,refd,cfg):
    n=len(pred);kt=refd['n_paths'];refs=[z[:k] for z,k in zip(refd['reference'],kt)];mask2=refd['observable_second_mask'];kp=np.array([len(z) for z in pred]);parts=np.zeros((n,5));sums=np.zeros((n,6));counts=np.zeros((n,6));second=np.zeros(n,int);cut=cfg['metric']['gospa_c']**2/2
    for qi,(a,b) in enumerate(zip(pred,refs)):
        e,pairs,miss,false=squared_gospa(a,b,cfg);param=0
        for i,j in pairs:
            param+=float(path_squared_distance(a[i],b[j],cfg));second[qi]+=int(mask2[qi,j]);delta=a[i]-b[j];delta[[2,4]]=(delta[[2,4]]+np.pi)%(2*np.pi)-np.pi;valid=np.ones(6,bool)
            for az in [2,4]:valid[az]=min(abs(np.sin(a[i,az+1])),abs(np.sin(b[j,az+1])))>cfg['metric']['azimuth_pole_sin_tolerance']
            sums[qi]+=np.where(valid,delta**2,0);counts[qi]+=valid
        parts[qi]=[e,param,cut*miss,cut*false,len(pairs)]
    groups={'all':np.ones(n,bool),'empty':kt==0,'nonempty':kt>0,'K1':kt==1,'K2plus':kt>=2,'second_related':mask2.any(-1)};rows=[];names=['gain_db','delay_ns','aoa_az_deg','aoa_zen_deg','aod_az_deg','aod_zen_deg']
    for name,mask in groups.items():
        ix=np.flatnonzero(mask);nk=int(kt[ix].sum());npred=int(kp[ix].sum());nr=int(parts[ix,4].sum());den2=int(mask2[ix].sum())
        if len(ix):
            rms=np.sqrt(np.divide(sums[ix].sum(0),counts[ix].sum(0),out=np.full(6,np.nan),where=counts[ix].sum(0)>0));rms[1]*=1e9;rms[2:]*=180/np.pi
            row=dict(subset=name,n_queries=len(ix),population_mass=len(ix)/n,global_gospa=float(np.sqrt(parts[ix,0].mean())),gospa_p90=float(np.sqrt(np.quantile(parts[ix,0],.9))),matched_parameter_cost=float(parts[ix,1].mean()),miss_cost=float(parts[ix,2].mean()),false_cost=float(parts[ix,3].mean()),matched_recall=nr/nk if nk else np.nan,precision=nr/npred if npred else np.nan,mean_miss=float((kt[ix]-parts[ix,4]).mean()),mean_false=float((kp[ix]-parts[ix,4]).mean()),n_matched=nr,reference_components=nk,predicted_components=npred,second_component_total=den2,second_component_matched=int(second[ix].sum()),second_matched_recall=float(second[ix].sum()/den2) if den2 else np.nan,count_accuracy=float(np.mean(kt[ix]==kp[ix])),**{'rmse_'+k:float(v) for k,v in zip(names,rms)},**{'n_'+k:int(v) for k,v in zip(names,counts[ix].sum(0))})
        else:row=dict(subset=name,n_queries=0,population_mass=0,global_gospa=np.nan,second_component_total=0,second_matched_recall=np.nan)
        rows.append(row)
    conf=np.zeros((4,4),int);np.add.at(conf,(kt,kp),1)
    z=np.zeros((n,3,6))
    for i,a in enumerate(pred):z[i,:len(a)]=a
    return rows,dict(decomposition=parts,prediction=z,predicted_k=kp,true_k=kt,confusion=conf,second_matched=second)

def run(phase,pair):
    cfg=config();torch.set_num_threads(2);ref=np.load(EXP/f'cache/reference_{phase}.npz');f=np.load(EXP/f'cache/features_{phase}.npz');root=EXP/f'runs/{phase}/{pair}';windows=2 if phase=='smoke' else 8
    for branch,variant in [('I-D','D'),('I-G','M3'),('C-G','M3'),('P-G','M3')]:
        m=make_model(cfg,pair,variant)
        for r in range(windows):
            dest=root/branch/f'evaluation_{r:02d}.npz';metric=root/branch/f'metrics_{r:02d}.json'
            if dest.exists() and metric.exists():continue
            s=torch.load(root/branch/f'checkpoints/update_{r:02d}.pt',map_location=m.device,weights_only=False);m.load_state_dict(s['committee']);before=time.perf_counter();z,l=raw_predict(m,f[variant]);pred,med=decode(z,l,.5,cfg,ids=True);inference=time.perf_counter()-before;before=time.perf_counter();rows,detail=score(pred,ref,cfg);detail['medoid']=med
            if r==windows-1:detail.update(raw_paths=z,logits=l)
            else:detail.update(diagnostic_raw_paths=z[:,:16],diagnostic_logits=l[:,:16])
            np.savez_compressed(dest,**detail)
            save(metric,dict(phase=phase,pair=pair,branch=branch,variant=variant,round=r,minutes=30*(r+1),n_attempts=1800*(r+1),steps=s['committee']['steps'],threshold=.5,inference_s=inference,scoring_s=time.perf_counter()-before,checkpoint_sha256=sha(root/branch/f'checkpoints/update_{r:02d}.pt'),metrics=rows))
            print('EVALUATED',phase,pair,branch,r,flush=True)
    save(root/'evaluation_complete.json',dict(phase=phase,pair=pair,status='complete',epoch=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','formal'],required=True);p.add_argument('--pair',type=int,required=True);a=p.parse_args()
    try:run(a.phase,a.pair)
    except Exception as e:failure('evaluate',e);raise
