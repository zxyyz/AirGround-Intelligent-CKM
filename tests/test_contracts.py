import itertools
import numpy as np
import pytest
import torch
from paths import EXP,VARIANTS,config
from models import SetModel
from data import features,raw_map,folds,subset_indices
from losses import set_loss
from candidate_level import CandidateLevelCKM,generalized_loss
from evaluate import gospa_batch
from ckm_sim.metrics import squared_gospa
from ckm_sim.geometry import GeometryMap
from train import setup,clip_independent

torch.set_num_threads(2)
def sample():return torch.tensor(features('dev','Full')[:4])
def labels(k):
 z=torch.zeros(4,3,6);z[:,:,0]=-100;z[:,:,1]=3e-7;z[:,:,3]=.7;z[:,:,5]=2.3
 return z,torch.tensor([k]*4)

@pytest.mark.parametrize('k',[0,1,2,3])
def test_original_forward_and_loss(k):
 model=SetModel('Full').initialize('test',17,0);old=CandidateLevelCKM(config())
 old.load_state_dict({k:v for k,v in model.state_dict().items() if k!='codes'})
 x=sample();out=model(x);z,length=labels(k)
 for a,b in zip(out[:5],old(x)):torch.testing.assert_close(a,b,rtol=0,atol=0)
 torch.testing.assert_close(set_loss(out,z,length),generalized_loss(out[:5],z,length,config()))

def test_input_separation_and_pooling():
 a=SetModel('Coord').initialize('test',1,0);x=sample();y=x.clone();y[:,6:]=torch.randn_like(y[:,6:])*100
 for p,q in zip(a(x),a(y)):torch.testing.assert_close(p,q,rtol=0,atol=0)
 assert a(x)[0][0].unique().numel()>1
 b=SetModel('RawMap').initialize('test',1,0);q=np.load(EXP/'cache/queries_dev.npz')['queries'][:4];g=GeometryMap(config())
 x=torch.tensor(raw_map(q));y=torch.tensor(raw_map(q,g.buildings[::-1].copy()))
 for p,q in zip(b(x),b(y)):torch.testing.assert_close(p,q,rtol=2e-5,atol=2e-5)
 assert x[:,6:].reshape(-1,64,40)[:,:16,:5].shape==(4,16,5)
 assert torch.count_nonzero(x[:,6:].reshape(-1,64,40)[:,16:])==0
 c=SetModel('GeoPool');assert [(n,p.shape) for n,p in b.named_parameters()]==[(n,p.shape) for n,p in c.named_parameters()]
 assert sum(p.numel() for p in b.parameters())==67954
 assert sum(p.numel() for p in SetModel('Full').parameters())==68498

def test_direct_only_changes_decode():
 a=SetModel('Full').initialize('test',2,0);b=SetModel('GeoDirect').initialize('test',2,0)
 for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
 x=sample();aa=a(x);bb=b(x)
 for i in [0,1,5]:torch.testing.assert_close(aa[i],bb[i],rtol=0,atol=0)
 assert not torch.allclose(aa[2],bb[2],atol=1e-10,rtol=1e-4)

@pytest.mark.parametrize('variant',VARIANTS)
def test_empty_and_insufficient_support(variant):
 model=SetModel(variant).initialize('test',1,0);x=torch.zeros(4,2566);out=model(x);z,k=labels(3);k=torch.arange(4)
 loss=set_loss(out,z,k);loss.backward();assert torch.isfinite(loss)
 assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 z[:,1:]=float('nan');k=torch.zeros(4,dtype=torch.long);assert torch.isfinite(set_loss(model(x),z,k))

def test_scalar_metric():
 rng=np.random.default_rng(17);a=rng.normal(size=(100,3,6));b=rng.normal(size=(100,3,6));a[:,:,0]-=100;b[:,:,0]-=100;a[:,:,1]*=1e-8;b[:,:,1]*=1e-8
 ak=rng.integers(0,4,100);bk=rng.integers(0,4,100);sq,m=gospa_batch(a,ak,b,bk)
 for i in range(100):
  expected,pairs,_,_=squared_gospa(a[i,:ak[i]],b[i,:bk[i]],config())
  assert sq[i]==pytest.approx(expected,abs=1e-11)
  assert [(j,int(k)) for j,k in enumerate(m[i]) if k>=0]==list(pairs)

def test_training_split_and_fresh_queries():
 from scipy.spatial import cKDTree
 p=np.load(EXP/'cache/pool.npz');assert not p['buffer_excluded'][p['train']].any()
 for pair in range(5101,5111):
  previous=set()
  for ratio in [.1,.25,.5,1]:
   ix=subset_indices(p,pair,ratio);assert previous<=set(ix);previous=set(ix)
  assert len(previous)==9556
 q=np.load(EXP/'cache/queries_confirm.npz')['queries'];assert len(q)==4096 and (folds(q)==4).all()
 train=np.unique(p['x'][p['train']],axis=0);tree=cKDTree(train[:,:3]);g=GeometryMap(config())
 for x,near in zip(q,tree.query_ball_point(q[:,:3],10.)):
  assert not near or not (g.road_distance(x[3:],train[near,3:])<=5.).any()

@pytest.mark.parametrize('device',['cpu','cuda:0','cuda:1'])
@pytest.mark.parametrize('variant',['RawMap','Full'])
def test_independent_updates(device,variant):
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 p,b,forward,_=setup('test',[17],variant,device);models=[SetModel(variant).initialize('test',17,i).to(device) for i in range(3)]
 opts=[torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001) for m in models];opt=torch.optim.AdamW(p.values(),lr=.001,weight_decay=.0001)
 x=torch.tensor(features('dev',variant)[:4],device=device);z,k=labels(3);z=z.to(device);k=k.to(device)
 for step in range(2):
  opt.zero_grad();out=forward(p,b,x[None].expand(3,-1,-1));loss=sum(set_loss(tuple(t[i] for t in out),z,k) for i in range(3));loss.backward();clip_independent(p);opt.step()
  for i,(model,op) in enumerate(zip(models,opts)):
   op.zero_grad();set_loss(model(x),z,k).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);op.step()
   for name,param in model.named_parameters():torch.testing.assert_close(p[name][i],param,rtol=3e-4,atol=3e-5)
