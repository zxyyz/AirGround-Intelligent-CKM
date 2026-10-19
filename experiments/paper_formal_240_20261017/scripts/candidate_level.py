"""Conditional M3. Public candidate paths; exact injection to noisy set labels.

For target cardinality<=3 an optimal injective assignment uses one of each
label's3 lowest candidate costs: at most2 choices can be occupied by others.
Enumerating3^3 distinct choices therefore solves the original injection loss
exactly for65 possible paths without factorial slot enumeration.
"""
import itertools,hashlib,time
import numpy as np,torch
from torch import nn
from ckm_sim.model import torch_direction_angle

def generalized_loss(out,labels,lengths,cfg):
 logit,gain,delay,aoa,aod=out;m=cfg['metric'];lp=cfg['model']['set_loss_lambda_p']
 def vec(a):return torch.stack([a[...,1].sin()*a[...,0].cos(),a[...,1].sin()*a[...,0].sin(),a[...,1].cos()],-1)
 distance=((gain[:,:,None]-labels[:,None,:,0])/m['gain_scale_db'])**2+((delay[:,:,None]-labels[:,None,:,1])/m['delay_scale_s'])**2
 for p,t,key in [(aoa,vec(labels[...,2:4]),'aoa_direction_scale_deg'),(aod,vec(labels[...,4:6]),'aod_direction_scale_deg')]:
  distance+=(torch_direction_angle(p[:,:,None,:].expand(-1,-1,3,-1),t[:,None,:,:].expand(-1,logit.shape[1],-1,-1))/np.deg2rad(m[key]))**2
 cost=distance-lp*logit[:,:,None];values,indices=cost.topk(3,dim=1,largest=False);comb=torch.tensor(list(itertools.product(range(3),repeat=3)),device=logit.device);cols=torch.arange(3,device=logit.device);chosen=indices[:,comb,cols];values=values[:,comb,cols];mask=cols[None,:]<lengths[:,None];invalid=torch.zeros_like(values[...,0],dtype=torch.bool)
 for a,b in [(0,1),(0,2),(1,2)]:invalid|=(chosen[...,a]==chosen[...,b])&(lengths[:,None]>b)
 best=(values*mask[:,None,:]).sum(-1).masked_fill(invalid,torch.inf).min(-1).values
 return (lp*nn.functional.softplus(logit).sum(-1)+best).mean()

class CandidateLevelCKM(nn.Module):
 def __init__(self,cfg):
  super().__init__();self.cfg=cfg;self.encoder=nn.Sequential(nn.Linear(40,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU());self.pathnet=nn.Sequential(nn.Linear(70,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,9));self.freenet=nn.Sequential(nn.Linear(134,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,9))
 def forward(self,f):
  c=f[:,6:].reshape(-1,64,40);valid=(c[...,0]>.5)&(c[...,9]>.5);e=self.encoder(c);pose=f[:,:6,None].transpose(1,2).expand(-1,64,-1);raw=self.pathnet(torch.cat([pose,e],-1));mask=valid[...,None];mean=(e*mask).sum(1)/mask.sum(1).clamp_min(1);maximum=e.masked_fill(~mask,-1e4).max(1).values;maximum=torch.where(valid.any(-1)[:,None],maximum,torch.zeros_like(maximum));free=self.freenet(torch.cat([f[:,:6],mean,maximum],-1))[:,None];gain=torch.cat([raw[...,1],free[...,1]],-1)*20-100;delay=torch.cat([(c[...,2]*self.cfg['propagation']['tau_max_s']+50e-9*raw[...,2].tanh()).clamp(0,self.cfg['propagation']['tau_max_s']),free[...,2].sigmoid()*self.cfg['propagation']['tau_max_s']],-1);logit=torch.cat([raw[...,0].masked_fill(~valid,-1e4),free[...,0]],-1);vectors=[]
  for start,col,sign in [(3,3,1),(6,6,-1)]:
   a=c[...,col:col+3]+.2*raw[...,start:start+3].tanh();a=torch.cat([a[...,:2],sign*a[...,2:].abs().clamp_min(1e-8)],-1);b=torch.cat([free[...,start:start+2],sign*nn.functional.softplus(free[...,start+2:start+3])],-1);v=torch.cat([a,b],1);vectors.append(v/torch.linalg.vector_norm(v,dim=-1,keepdim=True).clamp_min(1e-8))
  return logit,gain,delay,*vectors

def train_candidate_level(self,features,paths,steps):
 from learning import replay_weights
 n=len(paths);lens=np.array([len(p) for p in paths]);pad=np.zeros((n,3,6),np.float32)
 for i,p in enumerate(paths):pad[i,:len(p)]=p
 data=torch.as_tensor(features,device=self.device);labels=torch.as_tensor(pad,device=self.device);lengths=torch.as_tensor(lens,device=self.device);weights=torch.as_tensor(replay_weights(self.poses,lens,self.geometry),dtype=torch.float64);logs=[];t=self.cfg['training']
 for i,(model,opt,rng) in enumerate(zip(self.models,self.optimizers,self.samplers)):
  start=time.perf_counter();losses=[];norms=[];stream=hashlib.sha256();counts=np.zeros(4,int);model.train()
  for step in range(steps):
   idx=torch.multinomial(weights,64,replacement=True,generator=rng);stream.update(idx.numpy().tobytes());counts+=np.bincount(lens[idx.numpy()],minlength=4);idx=idx.to(self.device);opt.zero_grad(set_to_none=True);loss=generalized_loss(model(data[idx]),labels[idx],lengths[idx],self.cfg)
   if not torch.isfinite(loss):raise FloatingPointError('M3 nonfinite loss')
   loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),t['gradient_clip_norm']);opt.step();self.steps[i]+=1;losses.append(float(loss.detach()));norms.append(float(norm))
  logs.append(dict(model=i,steps_total=self.steps[i],n_train=n,loss_start=float(np.mean(losses[:20])),loss_end=float(np.mean(losses[-20:])),gradient_norm_mean=float(np.mean(norms)),sample_index_sha256=stream.hexdigest(),sampled_label_count_histogram=counts.tolist(),wall_s=time.perf_counter()-start,objective='exact injection over public candidates+1free, targetK<=3'))
 return logs
