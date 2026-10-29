"""Complete latency including online public geometry and ensemble decoding."""
import time,json
import numpy as np
import torch
from paths import EXP,VARIANTS,config,run_dir,save,stamp
from models import SetModel
from data import raw_map
from public_multibounce_geometry import PublicGeometry
from ckm_sim.geometry import angles
from evaluate import decode_members

def main():
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 freeze=json.loads((EXP/'configs/freeze.json').read_text());q=np.load(EXP/'cache/queries_dev.npz')['queries'][:128];geometry=PublicGeometry(config());rows=[]
 for v in VARIANTS:
  ck=torch.load(run_dir('formal',v,freeze['learning_rates'][v],5101)/f'step_{freeze["steps"]:06d}/checkpoint.pt',map_location='cpu',weights_only=True)
  models=[]
  for state in ck['states']:
   m=SetModel(v).to('cuda:0');m.load_state_dict(state);m.eval();models.append(m)
  records=[]
  for repeat in range(11):
   for x in q:
    torch.cuda.synchronize();start=time.perf_counter()
    if v=='RawMap':f=raw_map(x)
    elif v=='Coord':f=np.zeros((1,2566),np.float32);f[:,:6]=x/np.array([240,240,120,240,240,120])
    else:f,_=geometry.features(x[None],level=2)
    geo_done=time.perf_counter();f=torch.tensor(f,device='cuda:0');zz=[];ll=[]
    with torch.no_grad():
     for m in models:
      l,g,t,a,d=[z.cpu().numpy() for z in m(f)[:5]]
      z=np.concatenate([g[...,None],t[...,None],angles(a),angles(d)],-1);z[...,3]=np.minimum(z[...,3],np.pi-z[...,3]);z[...,5]=np.maximum(z[...,5],np.pi-z[...,5]);zz.append(z);ll.append(l)
    torch.cuda.synchronize();network_done=time.perf_counter();decode_members(np.array(zz),np.array(ll),freeze['thresholds'][v]);done=time.perf_counter()
    if repeat:records.append([(geo_done-start)*1000,(network_done-geo_done)*1000,(done-network_done)*1000,(done-start)*1000])
  a=np.array(records);row=dict(variant=v,parameter_count=sum(p.numel() for p in models[0].parameters()),device=torch.cuda.get_device_name(),queries=128,timed_passes=10,warmup_passes=1,mean_geometry_ms=float(a[:,0].mean()),mean_network_transfer_ms=float(a[:,1].mean()),mean_decoding_ms=float(a[:,2].mean()),mean_complete_ms=float(a[:,3].mean()),median_complete_ms=float(np.median(a[:,3])),p95_complete_ms=float(np.quantile(a[:,3],.95)))
  rows.append(row);print('LATENCY',row,flush=True)
 save(EXP/'results/inference_cost.json',dict(created_at=stamp(),scope='Single request, CPU feature generation + host/device transfers + three networks + NMS + float64 medoid; excludes external truth scoring',variants=rows))
if __name__=='__main__':main()
