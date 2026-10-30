import copy,json,time
import numpy as np
import torch
from paths import EXP,config,save
from ckm_sim.music import RFEstimator
from ckm_sim.sounding import simulate_received
from ckm_sim.metrics import squared_gospa
from revision_frontend import synth
from physics import fold
c=config();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
p=json.load(open(EXP/'configs/frontend_pilot_positions.json'));ids=p['indices'];records=[]
for file in sorted((EXP/'audit/reproduction_inputs/development_rays').glob('dev_*_rt.json')):records.extend(json.loads(file.read_text()))
cal=json.load(open(EXP/'configs/calibration.json'));c['runtime']['receiver_batch_size']=32;est=RFEstimator(c,cal);archive=np.load(EXP/'cache/frontend_pilot_revised.npz');rows=[];start=time.perf_counter()
for first in range(0,64,8):
 signals=[]
 for qi in ids[first:first+8]:
  h,_=synth(records[qi],c)
  for noise in range(4):s,pilot=simulate_received(h,(2026102910+noise,0,int(qi)),c,noiseless=noise==0);signals.append(s)
 zz,qq=est.estimate_many(np.stack(signals),pilot)
 for j,(z,q) in enumerate(zip(zz,qq),4*first):
  ref=archive['labels'][j,:archive['n_paths'][j]];cost=squared_gospa(fold(z),ref,c)[0];rows.append(dict(frame=j,batch_valid=q['valid'],scalar_K=len(ref),batch_K=len(z),squared_difference=cost))
summary=dict(passed=all(x['batch_valid'] and x['squared_difference']<=1e-4 for x in rows),effective_default_mode=config()['runtime'].get('receiver_batch_size',1),tested_batch_mode=32,maximum_squared_difference=max(x['squared_difference'] for x in rows),different_cardinalities=sum(x['scalar_K']!=x['batch_K'] for x in rows),wall_seconds=time.perf_counter()-start,rows=rows)
save(EXP/'results/frontend_batch32_check.json',summary);print({k:v for k,v in summary.items() if k!='rows'})
