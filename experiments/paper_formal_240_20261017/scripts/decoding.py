"""Prediction only; contains no query/reference files or evaluator access."""
import numpy as np
import torch
from ckm_sim.geometry import angles,directions
from ckm_sim.metrics import spherical_angle
from ckm_sim.model import committee_distances

def raw_predict(m,features,chunk=512):
    zs=[];ls=[]
    for model in m.models:
        model.eval();az=[];al=[]
        with torch.no_grad():
            for first in range(0,len(features),chunk):
                out=model(torch.as_tensor(features[first:first+chunk],device=m.device))
                l,g,t,a,d=[v.cpu().numpy() for v in out]
                z=np.concatenate([g[...,None],t[...,None],angles(a),angles(d)],-1)
                z[...,3]=np.minimum(z[...,3],np.pi-z[...,3]);z[...,5]=np.maximum(z[...,5],np.pi-z[...,5])
                az.append(z);al.append(l)
        zs.append(np.concatenate(az));ls.append(np.concatenate(al))
    return np.array(zs),np.array(ls)

def decode(z,l,threshold,cfg,ids=False,members_only=False):
    prob=1/(1+np.exp(-np.clip(l,-80,80)));members=[]
    for zz,pp in zip(z,prob):
        member=[]
        for row,p in zip(zz,pp):
            keep=np.flatnonzero(p>=threshold)
            if len(row)>3:
                selected=[]
                for idx in keep[np.argsort(-p[keep],kind='stable')]:
                    if any(abs(row[idx,1]-row[j,1])<=5e-9 and all(spherical_angle(directions(row[idx,a:a+2]),directions(row[j,a:a+2]))<=np.deg2rad(5) for a in [2,4]) for j in selected):continue
                    selected.append(idx)
                    if len(selected)==3:break
                keep=np.asarray(selected,dtype=int)
            keep=keep[np.argsort(-row[keep,0],kind='stable')];member.append(row[keep])
        members.append(member)
    if members_only:return members
    sums,_=committee_distances(members,cfg);med=sums.argmin(0);pred=[members[j][i] for i,j in enumerate(med)]
    return (pred,med) if ids else pred
