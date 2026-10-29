"""Three set predictors, exact logit-aware injection loss, squared-GOSPA medoid."""
import copy
import itertools
import numpy as np
import torch
from torch import nn
from .geometry import angles, directions
from .metrics import squared_gospa

class SetCKM(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg;m=cfg['model'];size=6+10*m['geometry_candidate_count']
        layers=[]
        for width in m['hidden_widths']:layers += [nn.Linear(size,width),nn.SiLU()];size=width
        layers.append(nn.Linear(size,9*cfg['ckm_target']['max_paths']));self.net=nn.Sequential(*layers)

    def forward(self,features):
        raw=self.net(features).reshape(-1,self.cfg['ckm_target']['max_paths'],9)
        logit=raw[...,0];gain=raw[...,1]*self.cfg['model']['gain_scale_db']+self.cfg['model']['gain_center_db']
        delay=torch.sigmoid(raw[...,2])*self.cfg['propagation']['tau_max_s']
        vectors=[]
        for j,sign in [(3,1),(6,-1)]:
            v=torch.cat([raw[...,j:j+2],sign*nn.functional.softplus(raw[...,j+2:j+3])],-1)
            vectors.append(v/torch.linalg.vector_norm(v,dim=-1,keepdim=True).clamp_min(1e-8))
        return logit,gain,delay,*vectors

def torch_direction_angle(u,v):
    return torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(u,v,dim=-1),dim=-1),
                       (u*v).sum(-1))

def matched_set_loss(output,labels,lengths,cfg):
    logit,gain,delay,aoa,aod=output;m=cfg['metric'];lp=cfg['model']['set_loss_lambda_p']
    def to_vec(a):
        return torch.stack([a[...,1].sin()*a[...,0].cos(),a[...,1].sin()*a[...,0].sin(),a[...,1].cos()],-1)
    la,ld=to_vec(labels[...,2:4]),to_vec(labels[...,4:6])
    distance=((gain[:,:,None]-labels[:,None,:,0])/m['gain_scale_db'])**2
    distance+=((delay[:,:,None]-labels[:,None,:,1])/m['delay_scale_s'])**2
    for predicted,target,key in [(aoa,la,'aoa_direction_scale_deg'),(aod,ld,'aod_direction_scale_deg')]:
        u=predicted[:,:,None,:].expand(-1,-1,labels.shape[1],-1)
        v=target[:,None,:,:].expand(-1,logit.shape[1],-1,-1)
        distance+=(torch_direction_angle(u,v)/np.deg2rad(m[key]))**2
    # All full permutations also enumerate each injection for a padded prefix.
    perms=torch.tensor(list(itertools.permutations(range(logit.shape[1]))),device=logit.device)
    columns=torch.arange(logit.shape[1],device=logit.device)
    cost=distance-lp*logit[:,:,None]
    assigned=cost[:,perms,columns]
    mask=columns[None,:]<lengths[:,None]
    best=(assigned*mask[:,None,:]).sum(-1).min(-1).values
    return (lp*nn.functional.softplus(logit).sum(-1)+best).mean()

def decode_path_set(output,cfg):
    logit,gain,delay,a,d=[v.detach().cpu().numpy() for v in output]
    z=np.concatenate([gain[...,None],delay[...,None],angles(a),angles(d)],-1)
    threshold=np.log(cfg['model']['probability_threshold']/(1-cfg['model']['probability_threshold']))
    return [row[keep][np.argsort(-row[keep,0],kind='stable')] for row,keep in zip(z,logit>=threshold)]

def committee_distances(predictions,cfg):
    m=len(predictions);n=len(predictions[0]);sums=np.zeros((m,n));total=np.zeros(n)
    for i in range(m):
        for j in range(i+1,m):
            d=np.array([squared_gospa(a,b,cfg)[0] for a,b in zip(predictions[i],predictions[j])])
            sums[i]+=d;sums[j]+=d;total+=d
    return sums,total*2/(m*(m-1))

def ensemble_medoid(predictions,cfg):
    sums,_=committee_distances(predictions,cfg);ids=sums.argmin(0)
    return [predictions[i][j] for j,i in enumerate(ids)]

class Committee:
    def __init__(self,cfg,pair_seed):
        self.cfg=cfg;self.device=cfg['runtime']['device'];self.models=[];self.optimizers=[];self.samplers=[]
        torch.set_num_threads(4)
        for i in range(cfg['model']['committee_size']):
            torch.manual_seed(pair_seed*100+10+i)
            model=SetCKM(cfg).to(self.device);self.models.append(model)
            t=cfg['training'];self.optimizers.append(torch.optim.AdamW(model.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay']))
            rng=torch.Generator().manual_seed(pair_seed*100+30+i);self.samplers.append(rng)
        self.steps=[0]*len(self.models)

    def clone(self):return copy.deepcopy(self)

    def train(self,features,paths,steps):
        k=self.cfg['ckm_target']['max_paths'];n=len(paths)
        if not n:raise ValueError('No valid observations')
        padded=np.zeros((n,k,6),np.float32);lengths=np.array([len(x) for x in paths])
        for i,p in enumerate(paths):padded[i,:len(p)]=p
        data=torch.as_tensor(features,device=self.device);labels=torch.as_tensor(padded,device=self.device)
        lengths=torch.as_tensor(lengths,device=self.device);batch=self.cfg['training']['logical_batch_size']
        logs=[]
        for i,(model,optimizer,rng) in enumerate(zip(self.models,self.optimizers,self.samplers)):
            model.train();losses=[]
            for _ in range(steps):
                idx=torch.randint(n,(batch,),generator=rng).to(self.device)
                optimizer.zero_grad(set_to_none=True);loss=matched_set_loss(model(data[idx]),labels[idx],lengths[idx],self.cfg)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),self.cfg['training']['gradient_clip_norm'])
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                optimizer.step();self.steps[i]+=1;losses.append(float(loss.detach().cpu()))
            logs.append({'model':i,'steps_total':self.steps[i],'loss_start':float(np.mean(losses[:min(20,len(losses))])),
                         'loss_end':float(np.mean(losses[-min(20,len(losses)):])),'n_train':n})
        return logs

    def predict_members(self,features):
        data=torch.as_tensor(features,device=self.device);result=[]
        for model in self.models:
            model.eval()
            with torch.no_grad():result.append(decode_path_set(model(data),self.cfg))
        return result

    def predict(self,features):return ensemble_medoid(self.predict_members(features),self.cfg)

    def state_dict(self):
        return {'models':[m.state_dict() for m in self.models],
                'optimizers':[o.state_dict() for o in self.optimizers],
                'samplers':[r.get_state() for r in self.samplers],'steps':self.steps}

    def load_state_dict(self,state):
        for m,v in zip(self.models,state['models']):m.load_state_dict(v)
        for o,v in zip(self.optimizers,state['optimizers']):o.load_state_dict(v)
        for r,v in zip(self.samplers,state['samplers']):r.set_state(v.cpu())
        self.steps=state['steps']
