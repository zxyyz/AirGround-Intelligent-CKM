"""Optional public-geometry anchor mixture plus bounded learned residual.

No traced path IDs, reference cardinality, material or power enter this model.
Candidate masks are public line-of-sight/finite-wall geometry. All-invalid rows
fall back to the original unconstrained set decoder. Presence and gain remain
measurement-trained, and the same injection loss/medoid apply.
"""
import numpy as np
import torch
from torch import nn

class GeometryResidualCKM(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg;m=cfg['model'];self.nc=m['geometry_candidate_count'];size=6+10*self.nc
        layers=[]
        for width in m['hidden_widths']:layers.extend([nn.Linear(size,width),nn.SiLU()]);size=width
        layers.append(nn.Linear(size,(9+self.nc)*cfg['ckm_target']['max_paths']));self.net=nn.Sequential(*layers)

    def forward(self,features):
        raw=self.net(features).reshape(-1,self.cfg['ckm_target']['max_paths'],9+self.nc)
        logit=raw[...,0];gain=raw[...,1]*self.cfg['model']['gain_scale_db']+self.cfg['model']['gain_center_db']
        candidates=features[:,6:].reshape(-1,self.nc,10)
        valid=(candidates[...,0]>.5)&(candidates[...,9]>.5)&(candidates[...,5]>=0)&(candidates[...,8]<=0)
        any_valid=valid.any(-1)
        selection=raw[...,9:].masked_fill(~valid[:,None,:],-1e4).softmax(-1)
        anchor_tau=torch.einsum('bkn,bn->bk',selection,candidates[...,2])*self.cfg['propagation']['tau_max_s']
        delta_tau=self.cfg['model'].get('delay_residual_bound_ns',50.)*1e-9*torch.tanh(raw[...,2])
        tau=(anchor_tau+delta_tau).clamp(0,self.cfg['propagation']['tau_max_s'])
        direct_tau=torch.sigmoid(raw[...,2])*self.cfg['propagation']['tau_max_s']
        tau=torch.where(any_valid[:,None],tau,direct_tau);vectors=[]
        for start,col,sign in [(3,3,1),(6,6,-1)]:
            anchor=torch.einsum('bkn,bnd->bkd',selection,candidates[...,col:col+3])
            v=anchor+self.cfg['model'].get('direction_residual_bound',.2)*torch.tanh(raw[...,start:start+3])
            v=torch.cat([v[...,:2],sign*v[...,2:].abs().clamp_min(1e-8)],-1)
            direct=torch.cat([raw[...,start:start+2],sign*nn.functional.softplus(raw[...,start+2:start+3])],-1)
            v=torch.where(any_valid[:,None,None],v,direct)
            vectors.append(v/torch.linalg.vector_norm(v,dim=-1,keepdim=True).clamp_min(1e-8))
        return logit,gain,tau,*vectors
