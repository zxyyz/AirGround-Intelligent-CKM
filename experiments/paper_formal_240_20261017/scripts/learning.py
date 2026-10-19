"""Minimal history-only learner ablations. No reference, evaluator or RT imports."""
from common import *
import numpy as np,torch
from torch import nn
from ckm_sim.v14_learning import LearningCommittee,loss_components
from ckm_sim.v14_residual import GeometryResidualCKM
from ckm_sim.model import matched_set_loss,decode_path_set
from ckm_sim.geometry import GeometryMap
from support import cell_keys

def canonical(z):
    z=np.array(z,dtype=float).reshape(-1,6).copy()
    z[:,3]=np.minimum(z[:,3],np.pi-z[:,3]);z[:,5]=np.maximum(z[:,5],np.pi-z[:,5]);return z

def valid_candidates(features,nc):
    c=features[:,6:].reshape(-1,nc,10)
    v=(c[...,0]>.5)&(c[...,9]>.5)&(c[...,5]>=0)&(c[...,8]<=0)
    return c,v

class HybridCKM(GeometryResidualCKM):
    """Reuse original anchor net; each sample/slot learns a geometry/free gate.

    Logit and gain stay shared and measurement-trained. Only delay/direction
    decode gains a free path. No fixed slot type or mechanism supervision.
    """
    def __init__(self,cfg):
        super().__init__(cfg);width=cfg['model']['hidden_widths'][-1];k=cfg['ckm_target']['max_paths']
        self.free_head=nn.Linear(width,7*k);self.gate_head=nn.Linear(width,k)
        nn.init.zeros_(self.gate_head.weight);nn.init.constant_(self.gate_head.bias,2.)
        self.force_gate=None;self.last_gate=None

    def forward(self,features):
        h=self.net[:-1](features);raw=self.net[-1](h).reshape(-1,3,9+self.nc)
        c,valid=valid_candidates(features,self.nc);any_valid=valid.any(-1)
        selection=raw[...,9:].masked_fill(~valid[:,None,:],-1e4).softmax(-1)
        anchor_tau=torch.einsum('bkn,bn->bk',selection,c[...,2])*self.cfg['propagation']['tau_max_s']
        tau=(anchor_tau+self.cfg['model'].get('delay_residual_bound_ns',50.)*1e-9*torch.tanh(raw[...,2])).clamp(0,self.cfg['propagation']['tau_max_s'])
        anchor_vec=[]
        for start,col,sign in [(3,3,1),(6,6,-1)]:
            a=torch.einsum('bkn,bnd->bkd',selection,c[...,col:col+3])
            v=a+self.cfg['model'].get('direction_residual_bound',.2)*torch.tanh(raw[...,start:start+3]);v=torch.cat([v[...,:2],sign*v[...,2:].abs().clamp_min(1e-8)],-1)
            anchor_vec.append(v/torch.linalg.vector_norm(v,dim=-1,keepdim=True).clamp_min(1e-8))
        free=self.free_head(h).reshape(-1,3,7);ft=torch.sigmoid(free[...,0])*self.cfg['propagation']['tau_max_s'];fv=[]
        for start,sign in [(1,1),(4,-1)]:
            v=torch.cat([free[...,start:start+2],sign*nn.functional.softplus(free[...,start+2:start+3])],-1)
            fv.append(v/torch.linalg.vector_norm(v,dim=-1,keepdim=True).clamp_min(1e-8))
        gate=torch.sigmoid(self.gate_head(h))
        if self.force_gate is not None:gate=torch.full_like(gate,self.force_gate)
        gate=torch.where(any_valid[:,None],gate,torch.zeros_like(gate));self.last_gate=gate
        vectors=[]
        for a,b in zip(anchor_vec,fv):
            v=gate[...,None]*a+(1-gate[...,None])*b
            vectors.append(v/torch.linalg.vector_norm(v,dim=-1,keepdim=True).clamp_min(1e-8))
        logit=raw[...,0];gain=raw[...,1]*self.cfg['model']['gain_scale_db']+self.cfg['model']['gain_center_db']
        return logit,gain,gate*tau+(1-gate)*ft,*vectors

def replay_weights(poses,lens,g):
    """Half uniform, half equal cells within K; preserves empirical K marginal."""
    n=len(lens);keys=cell_keys(poses,g,k=lens);groups={}
    for i,key in enumerate(keys):groups.setdefault(key,[]).append(i)
    nk=np.bincount(lens,minlength=4);nc=np.bincount([key[-1] for key in groups],minlength=4)
    balanced=np.zeros(n)
    for key,ix in groups.items():balanced[ix]=(nk[key[-1]]/n)/nc[key[-1]]/len(ix)
    weights=.5/n+.5*balanced;assert np.isclose(weights.sum(),1)
    for k in range(4):assert np.isclose(weights[lens==k].sum(),nk[k]/n)
    return weights

class Committee(LearningCommittee):
    def __init__(self,cfg,seed,spec='L0'):
        super().__init__(cfg,seed);self.spec=spec;self.poses=None;self.geometry=GeometryMap(cfg);self.last_free_grad=None
        if spec.startswith('Lf'):
            for i in range(3):
                # Old anchor weights are exactly reusable across architectures.
                old=self.models[i].state_dict();torch.manual_seed(seed*100+810+i)
                model=HybridCKM(cfg).to(self.device);model.load_state_dict(old,strict=False);self.models[i]=model
                self.optimizers[i]=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])

    def predict_members(self,features):return [[canonical(z) for z in member] for member in super().predict_members(features)]

    def probabilities(self,features):
        data=torch.as_tensor(features,device=self.device);out=[]
        for model in self.models:
            model.eval()
            with torch.no_grad():out.append(torch.sigmoid(model(data)[0]).cpu().numpy())
        return np.array(out)

    def gate_diagnostics(self,features):
        if not self.spec.startswith('Lf'):return {}
        data=torch.as_tensor(features,device=self.device);_,valid=valid_candidates(data,self.cfg['model']['geometry_candidate_count']);mask=valid.any(-1).cpu().numpy()
        gg=[]
        for model in self.models:
            with torch.no_grad():model(data);gg.append(model.last_gate.cpu().numpy())
        gates=np.array(gg)[:,mask]
        return dict(valid_geometry_slots=int(gates.size),free_gate_mean_valid=float((1-gates).mean()) if gates.size else None,
            valid_free_weight_gt_point1_fraction=float((gates<.9).mean()) if gates.size else 0.,
            valid_free_majority_fraction=float((gates<.5).mean()) if gates.size else 0.,
            valid_gate_gt_point99_fraction=float((gates>.99).mean()) if gates.size else 0.,free_gradient_norm=self.last_free_grad)

    def train(self,features,paths,steps):
        if self.spec=='L0':return super().train(features,paths,steps)
        n=len(paths);lens=np.array([len(p) for p in paths]);pad=np.zeros((n,3,6),np.float32)
        for i,p in enumerate(paths):pad[i,:len(p)]=p
        data=torch.as_tensor(features,device=self.device);labels=torch.as_tensor(pad,device=self.device);lengths=torch.as_tensor(lens,device=self.device)
        decay=('decay' in self.spec or self.spec in ['Ldecay','Lcombined']);replay=('replay' in self.spec or self.spec in ['Lreplay','Lcombined'])
        weights=None
        if replay:
            assert self.poses is not None and len(self.poses)==n
            weights=torch.as_tensor(replay_weights(self.poses,lens,self.geometry),dtype=torch.float64)
        logs=[];t=self.cfg['training']
        for i,(model,opt,rng) in enumerate(zip(self.models,self.optimizers,self.samplers)):
            model.train();losses=[];grads=[];terms=[];stream=hashlib.sha256();counts=np.zeros(4,int);start=time.perf_counter();free_grads=[]
            for step in range(steps):
                lr=t['learning_rate']*(.3+.7*(1+np.cos(np.pi*step/max(steps-1,1)))/2) if decay else t['learning_rate']
                for group in opt.param_groups:group['lr']=lr
                idx=torch.multinomial(weights,t['logical_batch_size'],replacement=True,generator=rng) if replay else torch.randint(n,(t['logical_batch_size'],),generator=rng)
                stream.update(idx.numpy().tobytes());counts+=np.bincount(lens[idx.numpy()],minlength=4);idx=idx.to(self.device)
                opt.zero_grad(set_to_none=True);out=model(data[idx]);loss=matched_set_loss(out,labels[idx],lengths[idx],self.cfg)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                if step%100==0 or step==steps-1:
                    with torch.no_grad():terms.append(dict(step=self.steps[i]+1,lr=lr,**loss_components(out,labels[idx],lengths[idx],self.cfg)))
                loss.backward()
                if self.spec.startswith('Lf') and (step%100==0 or step==steps-1):
                    free_grads.append(float(torch.linalg.vector_norm(model.free_head.weight.grad)))
                norm=nn.utils.clip_grad_norm_(model.parameters(),t['gradient_clip_norm'])
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                opt.step();self.steps[i]+=1;losses.append(float(loss.detach()));grads.append(float(norm))
            if free_grads:self.last_free_grad=float(np.mean(free_grads))
            logs.append(dict(model=i,steps_total=self.steps[i],n_train=n,loss_start=float(np.mean(losses[:20])),loss_end=float(np.mean(losses[-20:])),
                gradient_norm_mean=float(np.mean(grads)),lr_end=lr,sample_index_sha256=stream.hexdigest(),sampled_label_count_histogram=counts.tolist(),
                sampling='K-preserving mixture spatial replay' if replay else 'original uniform frame sampler',loss_components=terms,wall_s=time.perf_counter()-start,
                free_gradient_norm=float(np.mean(free_grads)) if free_grads else None))
        return logs

def calibration(probabilities,true_k):
    # Poisson-binomial count probabilities, averaged across the committee.
    n=probabilities.shape[1];dist=np.ones((3,n,1))
    for j in range(3):
        p=probabilities[:,:,j,None];new=np.zeros((*dist.shape[:2],dist.shape[-1]+1))
        new[:,:,:-1]+=dist*(1-p);new[:,:,1:]+=dist*p;dist=new
    dist=dist.mean(0);target=np.eye(4)[true_k]
    def ece(conf,correct):
        total=0.
        for lo in np.linspace(0,.9,10):
            keep=(conf>=lo)&(conf<lo+.1 if lo<.9 else conf<=1)
            if keep.any():total+=keep.mean()*abs(conf[keep].mean()-correct[keep].mean())
        return float(total)
    nonempty=1-dist[:,0];truth=(true_k>0).astype(float)
    return dict(count_brier=float(np.mean(np.sum((dist-target)**2,-1))),count_ECE=ece(dist.max(-1),(dist.argmax(-1)==true_k).astype(float)),
        nonempty_brier=float(np.mean((nonempty-truth)**2)),nonempty_ECE=ece(nonempty,truth))
