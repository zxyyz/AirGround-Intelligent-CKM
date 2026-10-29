"""Training-only memorization diagnostic for generic baselines."""
import numpy as np
import torch,time
from torch.func import vmap
from paths import EXP,save,stamp
from train import setup,clip_independent
from data import features
from losses import set_loss
from evaluate import decode_members,metrics
from ckm_sim.geometry import angles

def main():
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 p=np.load(EXP/'cache/pool.npz');pool=np.flatnonzero(p['train']&(p['pair']==5101)&(p['n_paths']>0));chosen=[];seen=set()
 for want in [1,2,3]:
  for i in pool:
   if p['n_paths'][i]==want and int(p['unique_index'][i]) not in seen:
    chosen.append(int(i));seen.add(int(p['unique_index'][i]))
    if sum(p['n_paths'][j]==want for j in chosen)==3:break
 ix=np.array(chosen);z=torch.tensor(p['labels'][ix],dtype=torch.float32,device='cuda:0')[None].expand(3,-1,-1,-1)
 k=torch.tensor(p['n_paths'][ix].astype(np.int64),device='cuda:0')[None].expand(3,-1);results=[]
 for variant in ['Coord','RawMap']:
  params,buffers,forward,count=setup('tiny_fit',[5101],variant,'cuda:0')
  f=torch.tensor(features('pool',variant)[p['unique_index'][ix]],device='cuda:0')[None].expand(3,-1,-1)
  objective=vmap(set_loss,in_dims=(0,0,0))
  def fn():return objective(forward(params,buffers,f),z,k)
  compiled=torch.compile(fn,fullgraph=True);opt=torch.optim.AdamW(params.values(),lr=.001,weight_decay=.0001);history=[];start=time.time()
  for step in range(5000):
   opt.zero_grad(set_to_none=True);loss=compiled();loss.sum().backward();clip_independent(params);opt.step()
   if step%500==0 or step==4999:
    value=float(loss.mean().detach());assert np.isfinite(value);history.append(dict(step=step+1,loss=value));print('TINY_FIT',variant,step+1,value,flush=True)
  with torch.no_grad():o=[a.cpu().numpy() for a in forward(params,buffers,f)[:5]]
  zz=np.concatenate([o[1][...,None],o[2][...,None],angles(o[3]),angles(o[4])],-1)
  pred,kk,_=decode_members(zz,o[0],.5);score,_=metrics(pred,kk,p['labels'][ix],p['n_paths'][ix],np.zeros((len(ix),3),bool))
  passed=history[-1]['loss']<.1*history[0]['loss'] and score['recall']>=.90
  results.append(dict(variant=variant,passed=passed,history=history,training_label_metrics=score,wall_s=time.time()-start))
 save(EXP/'results/tiny_fit.json',dict(passed=all(x['passed'] for x in results),frame_ids=p['ids'][ix].tolist(),label_source='training observations only',results=results,created_at=stamp()))
 assert all(x['passed'] for x in results),results
if __name__=='__main__':main()
