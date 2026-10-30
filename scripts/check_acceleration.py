"""Compare compiled and eager loss/gradients on the actual batched workload."""
import torch,time
from torch.func import vmap
from paths import EXP,save,stamp
from train import setup
from data import features
from losses import set_loss
import numpy as np

def main():
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 p=np.load(EXP/'cache/development_pool.npz');ix=np.flatnonzero(p['train']&(p['pair']==5101))[:64]
 z=torch.tensor(p['labels'][ix],dtype=torch.float32,device='cuda:0')[None].expand(9,-1,-1,-1)
 k=torch.tensor(p['n_paths'][ix].astype(np.int64),device='cuda:0')[None].expand(9,-1)
 results=[]
 for variant in ['Coord','RawMap','GeoPool','GeoDirect','Full']:
  params,buffers,forward,count=setup('acceleration',[5101,5102,5103],variant,'cuda:0')
  f=torch.tensor(features('pool',variant)[p['unique_index'][ix]],device='cuda:0')[None].expand(9,-1,-1)
  loss=vmap(set_loss,in_dims=(0,0,0))
  def fn(x,zz,kk):return loss(forward(params,buffers,x),zz,kk)
  eager=fn(f,z,k);eager.sum().backward();expected={n:v.grad.detach().clone() for n,v in params.items()}
  for v in params.values():v.grad=None
  compiled=torch.compile(fn,fullgraph=True);start=time.time();actual=compiled(f,z,k);actual.sum().backward();torch.cuda.synchronize()
  torch.testing.assert_close(actual,eager,rtol=3e-4,atol=3e-5)
  worst=0.
  for n,v in params.items():
   torch.testing.assert_close(v.grad,expected[n],rtol=3e-3,atol=3e-3)
   worst=max(worst,float((v.grad-expected[n]).abs().max()))
  results.append(dict(variant=variant,parameter_count=count,compile_and_first_backward_s=time.time()-start,maximum_absolute_gradient_difference=worst))
  print('ACCELERATION',variant,results[-1],flush=True)
 save(EXP/f'results/acceleration_{torch.cuda.get_device_name().replace(" ","_")}.json',dict(passed=True,device=torch.cuda.get_device_name(),variants=results,created_at=stamp()))
if __name__=='__main__':main()
