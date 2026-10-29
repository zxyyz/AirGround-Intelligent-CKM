"""Five learners; no observation references or material information."""
import math
import torch
from torch import nn
from paths import seed,VARIANTS

class SetModel(nn.Module):
    def __init__(self,variant='Full'):
        super().__init__();assert variant in VARIANTS;self.variant=variant
        pooled=variant in ['Coord','RawMap','GeoPool'];width=112 if pooled else 128
        pose_dim=54 if pooled else 6
        self.encoder=nn.Sequential(nn.Linear(40,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
        self.pathnet=nn.Sequential(nn.Linear(pose_dim+64,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,9))
        self.freenet=nn.Sequential(nn.Linear(pose_dim+128,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,9))
        phase=2*math.pi*(torch.arange(64)[:,None]+.5)*torch.arange(1,21)[None,:]/64
        self.register_buffer('codes',torch.cat([phase.sin(),phase.cos()],-1))

    def initialize(self,stage,pair,member):
        for name,p in self.named_parameters():
            rng=torch.Generator(device='cpu').manual_seed(seed('init',stage,pair,member,name))
            fan=p.shape[-1] if p.ndim==2 else dict(self.named_parameters())[name.replace('bias','weight')].shape[-1]
            with torch.no_grad():p.copy_(torch.empty(p.shape).uniform_(-1/math.sqrt(fan),1/math.sqrt(fan),generator=rng))
        return self

    def forward(self,f):
        pose=f[:,:6];c=f[:,6:].reshape(-1,64,40)
        if self.variant=='Coord':
            c=self.codes[None].expand(f.shape[0],-1,-1)
            valid=torch.ones(c.shape[:-1],dtype=torch.bool,device=f.device)
        elif self.variant=='RawMap':
            # All sixteen complete building tokens; remaining slots are padding.
            valid=torch.arange(64,device=f.device)[None,:].expand(f.shape[0],-1)<16
        else:valid=(c[...,0]>.5)&(c[...,9]>.5)
        e=self.encoder(c);mask=valid[...,None]
        mean=(e*mask).sum(1)/mask.sum(1).clamp_min(1)
        maximum=e.masked_fill(~mask,-1e4).max(1).values
        maximum=torch.where(valid.any(-1)[:,None],maximum,torch.zeros_like(maximum))
        if self.variant in ['Coord','RawMap','GeoPool']:
            code=nn.functional.pad(self.codes,(0,24))
            e=code[None]+.5*(mean+maximum)[:,None]
            valid=torch.ones_like(valid)
            phase=2*math.pi*pose[:,:,None]*torch.tensor([1.,2.,4.,8.],device=f.device)
            pose=torch.cat([pose,phase.sin().flatten(1),phase.cos().flatten(1)],-1)
        repeated=pose[:,None].expand(-1,64,-1)
        raw=self.pathnet(torch.cat([repeated,e],-1))
        free=self.freenet(torch.cat([pose,mean,maximum],-1))[:,None]
        anchored=self.variant=='Full'
        delay=(c[...,2]*1.8e-6+50e-9*raw[...,2].tanh()).clamp(0,1.8e-6) if anchored else raw[...,2].sigmoid()*1.8e-6
        logit=raw[...,0].masked_fill(~valid,-1e4);gain=raw[...,1]*20-100
        vectors=[]
        for start,col,sign in [(3,3,1),(6,6,-1)]:
            if anchored:
                a=c[...,col:col+3]+.2*raw[...,start:start+3].tanh()
                a=torch.cat([a[...,:2],sign*a[...,2:].abs().clamp_min(1e-8)],-1)
            else:a=torch.cat([raw[...,start:start+2],sign*nn.functional.softplus(raw[...,start+2:start+3])],-1)
            b=torch.cat([free[...,start:start+2],sign*nn.functional.softplus(free[...,start+2:start+3])],-1)
            vectors.append(nn.functional.normalize(torch.cat([a,b],1),dim=-1,eps=1e-8))
        return (torch.cat([logit,free[...,0]],-1),torch.cat([gain,free[...,1]*20-100],-1),
                torch.cat([delay,free[...,2].sigmoid()*1.8e-6],-1),*vectors,
                torch.cat([valid,torch.ones_like(valid[:,:1])],-1))
