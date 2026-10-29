"""Native differentiable objectives with exact assignment for K <= 3."""
import itertools, math
import torch
from torch.nn.functional import softplus


def parameter_distance(out,labels,lengths):
    logit,gain,delay,aoa,aod,*_=out
    present=torch.arange(3,device=labels.device)[None,:]<lengths[:,None]
    labels=torch.where(present[...,None],labels,torch.zeros_like(labels))
    d=((gain[:,:,None]-labels[:,None,:,0])/5)**2+((delay[:,:,None]-labels[:,None,:,1])/25e-9)**2
    for p,col in [(aoa,2),(aod,4)]:
        a=labels[...,col:col+2]
        target=torch.stack([a[...,1].sin()*a[...,0].cos(),a[...,1].sin()*a[...,0].sin(),a[...,1].cos()],-1)
        u,v=torch.broadcast_tensors(p[:,:,None,:],target[:,None,:,:])
        angle=torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(u,v,dim=-1),dim=-1),(u*v).sum(-1))
        d=d+(angle/math.radians(15))**2
    return d,present


def assignment(cost,present,partial=False):
    """Return optimal per-sample cost; top three suffice by an exchange argument."""
    values,indices=cost.topk(3,dim=1,largest=False)
    if partial:
        values=torch.cat([torch.zeros_like(values[:,:1]),values],1)
        indices=torch.cat([torch.full_like(indices[:,:1],-1),indices],1)
    n=4 if partial else 3
    combos=torch.tensor(list(itertools.product(range(n),repeat=3)),device=cost.device)
    cols=torch.arange(3,device=cost.device)
    chosen=indices[:,combos,cols]; selected=values[:,combos,cols]
    # where, rather than 0*inf, also allows genuinely hard inactive masks.
    selected=torch.where(present[:,None,:],selected,torch.zeros_like(selected))
    invalid=torch.zeros_like(selected[...,0],dtype=torch.bool)
    for a,b in [(0,1),(0,2),(1,2)]:
        invalid=invalid|((chosen[...,a]==chosen[...,b])&(chosen[...,a]>=0)&present[:,None,a]&present[:,None,b])
    totals=selected.sum(-1).masked_fill(invalid,torch.inf)
    best=totals.argmin(-1)
    choice=chosen.gather(1,best[:,None,None].expand(-1,1,3)).squeeze(1)
    return totals.gather(1,best[:,None]).squeeze(1),choice


def set_loss(out,labels,lengths,kind='ref',eta=.25,importance='uniform'):
    l=out[0]; d,present=parameter_distance(out,labels,lengths)
    if kind in ['ref','plain']:
        # Preserve historical soft masking, including its documented infeasible-support penalty.
        value,choice=assignment(d-l[:,:,None] if kind=='ref' else d,present)
        if kind=='plain':
            matched_logits=l.gather(1,choice.clamp_min(0))
            value=value-torch.where(present,matched_logits,torch.zeros_like(matched_logits)).sum(-1)
        return (softplus(l).sum(-1)+value).mean()
    raise ValueError('Only the original reference objective is available')
