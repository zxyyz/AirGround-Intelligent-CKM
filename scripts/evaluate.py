"""Float64 evaluation only. Independent reference never enters learner modules."""
import argparse, itertools, json, time
from pathlib import Path
import numpy as np
import torch
from paths import EXP, config, save, stamp
from models import SetModel
from data import features as load_features
from ckm_sim.geometry import angles, directions
from ckm_sim.metrics import path_squared_distance, spherical_angle


def gospa_batch(a,ak,b,bk):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    d=path_squared_distance(a[:,:,None,:],b[:,None,:,:],config())
    best=4.5*(ak+bk).astype(float);mapping=np.full((len(a),3),-1,np.int8)
    for n in range(1,4):
        for ai in itertools.combinations(range(3),n):
            for bj in itertools.permutations(range(3),n):
                valid=(ak>max(ai))&(bk>max(bj))
                val=4.5*(ak+bk-2*n)+d[:,ai,bj].sum(-1)
                improved=valid&(val<best-1e-12)
                best[improved]=val[improved];mapping[improved]=-1
                for i,j in zip(ai,bj):mapping[improved,i]=j
    return best,mapping


def decode_members(z,l,threshold):
    # Thresholding and NMS use stored float32 predictions, exactly as the model emits.
    members=[];counts=[]
    for zz,ll in zip(z,l):
        out=np.zeros((len(zz),3,6),np.float64);k=np.zeros(len(zz),np.int32)
        for qi,(row,logit) in enumerate(zip(zz,ll)):
            keep=np.flatnonzero(logit>=np.log(threshold/(1-threshold)));selected=[]
            for idx in keep[np.argsort(-logit[keep],kind='stable')]:
                if any(abs(row[idx,1]-row[j,1])<=5e-9 and all(spherical_angle(directions(row[idx,a:a+2]),directions(row[j,a:a+2]))<=np.deg2rad(5) for a in [2,4]) for j in selected):continue
                selected.append(idx)
                if len(selected)==3:break
            selected=np.array(selected,dtype=int)
            selected=selected[np.argsort(-row[selected,0],kind='stable')]
            k[qi]=len(selected);out[qi,:len(selected)]=row[selected]
        members.append(out);counts.append(k)
    sums=np.zeros((3,len(z[0])))
    for i,j in [(0,1),(0,2),(1,2)]:
        cost,_=gospa_batch(members[i],counts[i],members[j],counts[j]);sums[i]+=cost;sums[j]+=cost
    med=sums.argmin(0);qi=np.arange(len(med))
    return np.array(members)[med,qi],np.array(counts)[med,qi],med


def raw_prediction(checkpoint,features,device='cuda:0'):
    out_z=[];out_l=[];start=time.time()
    for state in checkpoint['states']:
        model=SetModel(**checkpoint['specification']).to(device);model.load_state_dict(state);model.eval();z=[];l=[]
        with torch.no_grad():
            for first in range(0,len(features),256):
                raw=model(torch.tensor(features[first:first+256],device=device))
                ll,g,t,a,d=[v.cpu().numpy() for v in raw[:5]]
                zz=np.concatenate([g[...,None],t[...,None],angles(a),angles(d)],-1)
                zz[...,3]=np.minimum(zz[...,3],np.pi-zz[...,3]);zz[...,5]=np.maximum(zz[...,5],np.pi-zz[...,5])
                z.append(zz);l.append(ll)
        out_z.append(np.concatenate(z));out_l.append(np.concatenate(l))
    return np.array(out_z),np.array(out_l),time.time()-start


def metrics(pred,k,ref,rk,second):
    sq,mapping=gospa_batch(pred,k,ref,rk);match=(mapping>=0);nm=match.sum(-1);miss=rk-nm;false=k-nm
    result=dict(gospa=float(np.sqrt(sq.mean())),miss=float(miss.mean()),false=float(false.mean()),recall=float(nm.sum()/max(1,rk.sum())),precision=float(nm.sum()/max(1,k.sum())),K_mae=float(abs(k-rk).mean()),predicted_empty_fraction=float((k==0).mean()),reference_empty_fraction=float((rk==0).mean()),empty_false_path_rate=float((k[rk==0]>0).mean()) if (rk==0).any() else None)
    power_counts=np.zeros(3,int);power_matched=np.zeros(3,int);second_match=0;errors=[]
    for i in range(len(k)):
        matched_truth=set(mapping[i,match[i]].tolist())
        for j in range(rk[i]):
            band=0 if ref[i,j,0]<-105 else (1 if ref[i,j,0]<-95 else 2)
            power_counts[band]+=1;power_matched[band]+=j in matched_truth
            second_match+=bool(second[i,j] and j in matched_truth)
        for a,b in enumerate(mapping[i]):
            if b<0:continue
            delta=pred[i,a]-ref[i,b];delta[[2,4]]=(delta[[2,4]]+np.pi)%(2*np.pi)-np.pi
            for az in [2,4]:
                if min(abs(np.sin(pred[i,a,az+1])),abs(np.sin(ref[i,b,az+1])))<=.001:delta[az]=np.nan
            errors.append(delta)
    if errors:
        e=np.array(errors);rmse=np.sqrt(np.nanmean(e**2,axis=0));rmse[1]*=1e9;rmse[2:]*=180/np.pi
    else:rmse=[None]*6
    result.update({f'rmse_{key}':float(v) if v is not None else None for key,v in zip(['gain_db','delay_ns','aoa_az_deg','aoa_zen_deg','aod_az_deg','aod_zen_deg'],rmse)})
    result['power_counts']=power_counts.tolist();result['power_matched']=power_matched.tolist();result['second_reference_count']=int(second.sum());result['second_matched']=int(second_match)
    result['by_reference_K']={str(j):dict(n=int((rk==j).sum()),gospa=float(np.sqrt(sq[rk==j].mean())) if (rk==j).any() else None,miss=float(miss[rk==j].mean()) if (rk==j).any() else None,false=float(false[rk==j].mean()) if (rk==j).any() else None) for j in range(4)}
    return result,dict(prediction=pred,predicted_K=k,squared_gospa=sq,matching=mapping,miss=miss,false=false,reference_K=rk)


def evaluate(path,dataset,thresholds):
    path=Path(path);done=path/f'evaluation_{dataset}.json'
    if done.exists():return json.loads(done.read_text())
    ck=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)
    if dataset!='dev':
        freeze=json.loads((EXP/'configs/freeze.json').read_text());assert freeze['frozen']
    features=load_features(dataset,ck['name'])
    reference=np.load(EXP/f'cache/reference_{dataset}.npz')
    raw,l,forward_s=raw_prediction(ck,features)
    scores=[];start=time.time()
    for threshold in thresholds:
        pred,k,med=decode_members(raw,l,threshold)
        score,arrays=metrics(pred,k,reference['reference'],reference['n_paths'],reference['observable_second_mask'])
        score['threshold']=threshold;scores.append(score)
        np.savez_compressed(path/f'predictions_{dataset}_{threshold:.2f}.npz',**arrays,medoid_member=med)
    result=dict(dataset=dataset,name=ck['name'],pair=ck['pair'],ratio=ck['ratio'],steps=ck['steps'],lr=ck['lr'],scores=scores,forward_s=forward_s,decode_and_evaluation_s=time.time()-start,metric='float64_both_inputs_v1',created_at=stamp())
    save(done,result);print('EVALUATED',path,dataset,[(x['threshold'],round(x['gospa'],5)) for x in scores],flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint_dir');p.add_argument('--dataset',choices=['dev','confirm','domain'],default='dev');p.add_argument('--thresholds',default='.25,.35,.5,.65,.75');a=p.parse_args();torch.set_num_threads(2)
    evaluate(a.checkpoint_dir,a.dataset,list(map(float,a.thresholds.split(','))))
