"""Auditable uniform-history learning; optional fixed per-update LR decay."""
import hashlib
import itertools
import time
import numpy as np
import torch
from torch import nn
from .model import Committee,matched_set_loss,torch_direction_angle

def loss_components(output,labels,lengths,cfg):
    """Exact same winning injection as Eq.12; diagnostics do not affect gradients."""
    logit,gain,delay,aoa,aod=output;m=cfg['metric'];lp=cfg['model']['set_loss_lambda_p']
    def vec(a):return torch.stack([a[...,1].sin()*a[...,0].cos(),a[...,1].sin()*a[...,0].sin(),a[...,1].cos()],-1)
    components=[((gain[:,:,None]-labels[:,None,:,0])/m['gain_scale_db'])**2,
                ((delay[:,:,None]-labels[:,None,:,1])/m['delay_scale_s'])**2]
    for u,v,key in [(aoa,vec(labels[...,2:4]),'aoa_direction_scale_deg'),(aod,vec(labels[...,4:6]),'aod_direction_scale_deg')]:
        components.append((torch_direction_angle(u[:,:,None,:].expand(-1,-1,labels.shape[1],-1),
            v[:,None,:,:].expand(-1,logit.shape[1],-1,-1))/np.deg2rad(m[key]))**2)
    perms=torch.tensor(list(itertools.permutations(range(logit.shape[1]))),device=logit.device)
    cols=torch.arange(logit.shape[1],device=logit.device);mask=cols[None,:]<lengths[:,None]
    cost=sum(components)-lp*logit[:,:,None]
    best=(cost[:,perms,cols]*mask[:,None,:]).sum(-1).argmin(-1)
    row=torch.arange(len(labels),device=logit.device)[:,None];slots=perms[best]
    terms=[(c[row,slots,cols]*mask).sum(-1).mean() for c in components]
    presence=lp*(nn.functional.softplus(logit).sum(-1)-(logit[row,slots]*mask).sum(-1)).mean()
    return {k:float(v.detach()) for k,v in zip(['presence','gain','delay','aoa','aod'],[presence,*terms])}

class LearningCommittee(Committee):
    def __init__(self,cfg,pair_seed):
        super().__init__(cfg,pair_seed)
        if cfg['model'].get('variant')=='public_geometry_residual':
            from .v14_residual import GeometryResidualCKM
            self.models=[];self.optimizers=[]
            for i in range(cfg['model']['committee_size']):
                torch.manual_seed(pair_seed*100+10+i)
                model=GeometryResidualCKM(cfg).to(self.device);self.models.append(model)
                self.optimizers.append(torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay']))

    def train(self,features,paths,steps):
        n=len(paths);k=self.cfg['ckm_target']['max_paths']
        if not n or steps<1:raise ValueError('Empty training data or steps')
        padded=np.zeros((n,k,6),np.float32);lens=np.array([len(p) for p in paths])
        for i,p in enumerate(paths):padded[i,:len(p)]=p
        data=torch.as_tensor(features,device=self.device);labels=torch.as_tensor(padded,device=self.device)
        lengths=torch.as_tensor(lens,device=self.device);t=self.cfg['training'];batch=t['logical_batch_size'];logs=[]
        for i,(model,opt,rng) in enumerate(zip(self.models,self.optimizers,self.samplers)):
            model.train();losses=[];grads=[];terms=[];counts=np.zeros(k+1,np.int64);stream=hashlib.sha256();start=time.perf_counter()
            for step in range(steps):
                lr=t['learning_rate']
                total=getattr(self,'schedule_total_steps',steps);offset=getattr(self,'schedule_step_offset',0)
                if t.get('lr_schedule','constant')=='cosine_per_update':lr*=.1+.9*(1+np.cos(np.pi*(step+offset)/max(total-1,1)))/2
                for group in opt.param_groups:group['lr']=lr
                idx=torch.randint(n,(batch,),generator=rng);stream.update(idx.numpy().tobytes())
                counts+=np.bincount(lens[idx.numpy()],minlength=k+1)
                idx=idx.to(self.device);opt.zero_grad(set_to_none=True);out=model(data[idx]);loss=matched_set_loss(out,labels[idx],lengths[idx],self.cfg)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite learning loss')
                if step%100==0 or step==steps-1:
                    with torch.no_grad():terms.append({'step':self.steps[i]+1,'lr':lr,**loss_components(out,labels[idx],lengths[idx],self.cfg)})
                loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),t['gradient_clip_norm'])
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite learning gradient')
                opt.step();self.steps[i]+=1;losses.append(float(loss.detach()));grads.append(float(norm))
            logs.append({'model':i,'steps_total':self.steps[i],'n_train':n,'loss_start':float(np.mean(losses[:20])),
                'loss_end':float(np.mean(losses[-20:])),'gradient_norm_mean':float(np.mean(grads)),'gradient_norm_max':max(grads),
                'lr_end':lr,'sampled_label_count_histogram':counts.tolist(),'sample_index_sha256':stream.hexdigest(),
                'sampling':'uniform with replacement over valid prefix; same initialization and sampler seeds per setting',
                'loss_components':terms,'wall_s':time.perf_counter()-start})
        return logs
