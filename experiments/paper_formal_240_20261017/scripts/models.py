"""Measurement-set supervision only. Geometry encoder has no RT/evaluator inputs."""
from common import *
import numpy as np,torch
from torch import nn
from learning import Committee,HybridCKM
from public_multibounce_geometry import D
class CandidateSetCKM(nn.Module):
 def __init__(self,cfg):
  super().__init__();self.cfg=cfg;self.nc=64
  self.encoder=nn.Sequential(nn.Linear(D,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
  self.net=nn.Sequential(nn.Linear(134,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,3*(9+64)))
  self.free_head=nn.Linear(128,21);self.gate_head=nn.Linear(128,3);nn.init.zeros_(self.gate_head.weight);nn.init.constant_(self.gate_head.bias,2.);self.last_gate=None
 def forward(self,features):
  c=features[:,6:].reshape(-1,64,D);valid=(c[...,0]>.5)&(c[...,9]>.5);anyvalid=valid.any(-1);emb=self.encoder(c);mask=valid[...,None];mean=(emb*mask).sum(1)/mask.sum(1).clamp_min(1);maximum=emb.masked_fill(~mask,-1e4).max(1).values;maximum=torch.where(anyvalid[:,None],maximum,torch.zeros_like(maximum));context=torch.cat([features[:,:6],mean,maximum],-1);h=self.net[:-1](context);raw=self.net[-1](h).reshape(-1,3,73);scores=torch.einsum('bkd,bnd->bkn',raw[...,9:],emb)/8;selection=scores.masked_fill(~valid[:,None,:],-1e4).softmax(-1)
  tau=(torch.einsum('bkn,bn->bk',selection,c[...,2])*self.cfg['propagation']['tau_max_s']+50e-9*torch.tanh(raw[...,2])).clamp(0,self.cfg['propagation']['tau_max_s']);free=self.free_head(h).reshape(-1,3,7);ft=free[...,0].sigmoid()*self.cfg['propagation']['tau_max_s'];gate=self.gate_head(h).sigmoid();gate=torch.where(anyvalid[:,None],gate,torch.zeros_like(gate));self.last_gate=gate;vectors=[]
  for start,col,fi,sign in [(3,3,1,1),(6,6,4,-1)]:
   a=torch.einsum('bkn,bnd->bkd',selection,c[...,col:col+3])+.2*torch.tanh(raw[...,start:start+3]);a=torch.cat([a[...,:2],sign*a[...,2:].abs().clamp_min(1e-8)],-1);a=a/torch.linalg.vector_norm(a,dim=-1,keepdim=True).clamp_min(1e-8);f=torch.cat([free[...,fi:fi+2],sign*nn.functional.softplus(free[...,fi+2:fi+3])],-1);f=f/torch.linalg.vector_norm(f,dim=-1,keepdim=True).clamp_min(1e-8);v=gate[...,None]*a+(1-gate[...,None])*f;vectors.append(v/torch.linalg.vector_norm(v,dim=-1,keepdim=True).clamp_min(1e-8))
  return raw[...,0],raw[...,1]*self.cfg['model']['gain_scale_db']+self.cfg['model']['gain_center_db'],gate*tau+(1-gate)*ft,*vectors
class ExperimentCommittee(Committee):
 def __init__(self,cfg,seed,variant):
  cfg=copy.deepcopy(cfg)
  if variant=='Mwide':cfg['model']['hidden_widths']=[144]*3
  super().__init__(cfg,seed,'Lf_replay');self.variant=variant
  if variant in ['M1','M2','M3']:
   self.models=[];self.optimizers=[]
   for i in range(3):
    if variant=='M3':
     from candidate_level import CandidateLevelCKM
     factory=CandidateLevelCKM
    else:factory=CandidateSetCKM
    torch.manual_seed(seed*100+810+i);model=factory(cfg).to(self.device);self.models.append(model);self.optimizers.append(torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay']))
 def train(self,features,paths,steps):
  if self.variant=='M3':
   from candidate_level import train_candidate_level
   return train_candidate_level(self,features,paths,steps)
  return super().train(features,paths,steps)
 def predict_members(self,features):
  if self.variant=='M3':
   from decoding import raw_predict,decode
   z,l=raw_predict(self,features)
   return decode(z,l,self.cfg['model']['probability_threshold'],self.cfg,members_only=True)
  return super().predict_members(features)
