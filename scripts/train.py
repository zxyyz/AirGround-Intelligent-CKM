"""Paired independent optimization with exact resume and immutable data indices."""
import argparse,time,json,os
import numpy as np
import torch
from torch.func import functional_call,stack_module_state,vmap
from paths import EXP,VARIANTS,config,seed,save,sha,stamp,run_dir
from models import SetModel
from losses import set_loss
from data import features,subset_indices,blocks

def measurement_pair(stage,pair):return 5101+(pair-2026102701) if stage=='pilot' else pair

def sampling_weights(p,ix):
    k=p['n_paths'][ix];groups={}
    for j,b in enumerate(blocks(p['x'][ix])):groups.setdefault((*b,int(k[j])),[]).append(j)
    nk=np.bincount(k,minlength=4);nc=np.bincount([key[-1] for key in groups],minlength=4)
    weight=np.full(len(ix),.5/len(ix))
    for key,j in groups.items():weight[j]+=.5*(nk[key[-1]]/len(ix))/nc[key[-1]]/len(j)
    assert np.isclose(weight.sum(),1);return weight

def stream(stage,pair,ratio,steps,p):
    path=EXP/f'manifests/batches/{stage}_{pair}_{ratio:.2f}.npz'
    if path.exists():
        data=np.load(path);assert data['indices'].shape[0]==3 and data['indices'].shape[1]>=steps
        return data['indices'][:,:steps],data['subset']
    ix=subset_indices(p,measurement_pair(stage,pair),ratio);w=sampling_weights(p,ix);batches=[]
    for member in range(3):
        rng=np.random.Generator(np.random.PCG64DXSM(seed('sampling',stage,pair,member,ratio)))
        batches.append(ix[rng.choice(len(ix),(steps,64),replace=True,p=w)])
    path.parent.mkdir(parents=True,exist_ok=True);data=np.array(batches,np.int32)
    np.savez_compressed(path,indices=data,subset=ix)
    save(path.with_suffix('.json'),dict(sha256=sha(path),stage=stage,pair=pair,ratio=ratio,steps=steps,subset_count=len(ix),unique_positions=len(np.unique(p['unique_index'][ix])),created_at=stamp()))
    return data,ix

def setup(stage,pairs,variant,device):
    models=[SetModel(variant).initialize(stage,pair,m) for pair in pairs for m in range(3)]
    params,buffers=stack_module_state(models)
    params={k:torch.nn.Parameter(v.to(device)) for k,v in params.items()};buffers={k:v.to(device) for k,v in buffers.items()}
    base=models[0].to('meta')
    def forward(p,b,x):return functional_call(base,(p,b),(x,))
    return params,buffers,vmap(forward,in_dims=(0,0,0)),sum(p.numel() for p in models[-1].parameters())

def clip_independent(params,max_norm=5):
    grads=[p.grad for p in params.values() if p.grad is not None]
    norm=torch.sqrt(sum(g.flatten(1).square().sum(-1) for g in grads))
    factor=(max_norm/(norm+1e-6)).clamp_max(1)
    for g in grads:g.mul_(factor.reshape(-1,*([1]*(g.ndim-1))))
    return norm

def atomic_torch_save(obj,path):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');torch.save(obj,tmp);tmp.replace(path)
    if path.name.startswith('resume_'):save(path.with_suffix('.sha256.json'),dict(sha256=sha(path),step=obj['step']))

def train(stage,pairs,variant,steps,ratio=1.,lr=.001,compile_model=True):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if stage=='formal':
        freeze=json.loads((EXP/'configs/freeze.json').read_text());assert freeze['frozen'] and steps==freeze['steps']
        assert lr==freeze['learning_rates'][variant]
        for path,h in freeze['training_source_hashes'].items():assert sha(EXP/path)==h,path
    device='cuda:0';source=np.load(EXP/'cache/pool.npz');p={k:source[k] for k in source.files}
    dest=run_dir(stage,variant,lr,pairs[0],ratio).parent;dest.mkdir(parents=True,exist_ok=True)
    marker=dest/f'group_{pairs[0]}_{steps}_complete.json'
    if marker.exists():return
    f=torch.tensor(features('pool',variant),device=device);pos=torch.tensor(p['unique_index'],device=device)
    z=torch.tensor(p['labels'],dtype=torch.float32,device=device);k=torch.tensor(p['n_paths'].astype(np.int64),device=device)
    batches=[];counts={}
    for pair in pairs:
        batch,ix=stream(stage,pair,ratio,steps,p);batches.extend(batch);counts[pair]=len(ix)
    batches=torch.tensor(np.array(batches).transpose(1,0,2).astype(np.int64),device=device)
    params,buffers,forward,count=setup(stage,pairs,variant,device)
    opt=torch.optim.AdamW(params.values(),lr=lr,weight_decay=.0001,foreach=True)
    state_file=dest/f'resume_{pairs[0]}.pt';start_step=0;history=[];prior_wall=0.
    if state_file.exists():
        integrity=state_file.with_suffix('.sha256.json')
        if integrity.exists():assert sha(state_file)==json.loads(integrity.read_text())['sha256']
        elif stage=='formal':raise RuntimeError('Formal resume integrity record missing')
        state=torch.load(state_file,map_location=device,weights_only=True)
        assert state['pairs']==pairs and state['variant']==variant and state['lr']==lr
        start_step=state['step'];history=state['history'];prior_wall=state['wall_s']
        with torch.no_grad():
            for n,v in params.items():v.copy_(state['params'][n])
        opt.load_state_dict(state['optimizer']);del state
    objective=vmap(lambda out,zz,kk:set_loss(out,zz,kk),in_dims=(0,0,0))
    def computation(ff,zz,kk):return objective(forward(params,buffers,ff),zz,kk)
    if compile_model:computation=torch.compile(computation,fullgraph=True)
    start=time.time();torch.cuda.reset_peak_memory_stats();last_losses=None
    def snapshot(step):
        torch.cuda.synchronize();wall=prior_wall+time.time()-start
        atomic_torch_save(dict(step=step,pairs=pairs,variant=variant,lr=lr,params={n:v.detach().cpu() for n,v in params.items()},optimizer=opt.state_dict(),history=history,wall_s=wall),state_file)
        for group,pair in enumerate(pairs):
            states=[]
            for member in range(3):
                i=group*3+member;states.append({**{n:v[i].detach().cpu() for n,v in params.items()},**{n:v[i].detach().cpu() for n,v in buffers.items()}})
            out=run_dir(stage,variant,lr,pair,ratio)/f'step_{step:06d}'
            atomic_torch_save(dict(states=states,specification={'variant':variant},name=variant,stage=stage,pair=pair,steps=step,ratio=ratio,lr=lr),out/'checkpoint.pt')
            save(out/'training.json',dict(status='complete',checkpoint_sha256=sha(out/'checkpoint.pt'),pairs_batched=pairs,batched_wall_s=wall,amortized_wall_s=wall/len(pairs),peak_memory_bytes=torch.cuda.max_memory_allocated(),device=torch.cuda.get_device_name(),parameter_count=count,subset_count=counts[pair],steps_per_member=step,precision='float32; TF32 disabled',compile=compile_model,history=[{**h,'loss':h['loss'][group*3:group*3+3],'gradient_norm':h['gradient_norm'][group*3:group*3+3]} for h in history],finished_at=stamp()))
    if start_step==steps:snapshot(steps)
    for step in range(start_step,steps):
        idx=batches[step];opt.zero_grad(set_to_none=True)
        losses=computation(f[pos[idx]],z[idx],k[idx]);losses.sum().backward();norm=clip_independent(params);opt.step()
        if step%500==0 or step+1==steps:
            ls=losses.detach().cpu().numpy();ns=norm.detach().cpu().numpy()
            if not np.isfinite(ls).all() or not np.isfinite(ns).all():raise FloatingPointError(f'Nonfinite {stage} {variant} {lr} {step}')
            history.append(dict(step=step+1,loss=ls.tolist(),gradient_norm=ns.tolist(),wall_s=prior_wall+time.time()-start))
            save(dest/f'progress_{pairs[0]}.json',dict(pairs=pairs,step=step+1,steps=steps,history=history[-3:],variant=variant,lr=lr,device=torch.cuda.get_device_name()))
            print('TRAIN',stage,variant,lr,pairs,step+1,steps,round(float(ls.mean()),4),round(time.time()-start,1),flush=True)
        if step+1 in [7100,14200,28400,56800,steps]:snapshot(step+1)
        elif (step+1)%5000==0:
            atomic_torch_save(dict(step=step+1,pairs=pairs,variant=variant,lr=lr,params={n:v.detach().cpu() for n,v in params.items()},optimizer=opt.state_dict(),history=history,wall_s=prior_wall+time.time()-start),state_file)
    save(marker,dict(pairs=pairs,variant=variant,steps=steps,completed_at=stamp()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--pairs',required=True);p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--steps',type=int,required=True);p.add_argument('--ratio',type=float,default=1);p.add_argument('--lr',type=float,default=.001);p.add_argument('--eager',action='store_true');a=p.parse_args()
    train(a.stage,list(map(int,a.pairs.split(','))),a.variant,a.steps,a.ratio,a.lr,not a.eager)
