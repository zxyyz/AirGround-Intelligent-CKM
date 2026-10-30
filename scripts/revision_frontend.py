"""E2: preselected development-domain pilot and independent frequency contract."""
import copy,json,time,resource,subprocess
import numpy as np
import torch
from paths import EXP,config,save,sha,stamp
from ckm_sim.sounding import channel_from_paths,simulate_received,joint_atom,frequency_offsets,array_response
from ckm_sim.music import RFEstimator
from ckm_sim.metrics import squared_gospa
from physics import fold
from revision_metrics import csvout

def synth(rec,cfg):
 raw=[rec['paths'][j] for j in rec['signal_indices']]
 z=np.array([p['z'] for p in raw]).reshape(-1,6)
 alpha=np.array([complex(*p['coefficient']) for p in raw])*np.exp(-2j*np.pi*cfg['propagation']['carrier_hz']*z[:,1])
 assert np.isfinite(z).all() and np.isfinite(alpha).all()
 return channel_from_paths(z,alpha,cfg),fold(z[z[:,0]>=-110][:3])

def main():
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 start=time.perf_counter();cfg=config();old=EXP/'audit/reproduction_inputs';oldcfg=json.loads((old/'historical_configs/effective_config.json').read_text());oldcfg['runtime']['device']='cuda:0'
 q=np.load(EXP/'cache/queries_dev.npz')['queries'];rng=np.random.Generator(np.random.PCG64DXSM(2026102964));ids=np.sort(rng.choice(len(q),64,replace=False))
 save(EXP/'configs/frontend_pilot_positions.json',dict(seed=2026102964,indices=ids,queries=q[ids],query_file_sha256=sha(EXP/'cache/queries_dev.npz'),selected_before_frontend=True))
 rtroot=EXP/'audit/reproduction_inputs/development_rays';records=[]
 for path in sorted(rtroot.glob('dev_*_rt.json')):records.extend(json.loads(path.read_text()))
 assert len(records)==512
 for i in ids:np.testing.assert_allclose(records[i]['position'],q[i],rtol=0,atol=0);assert all('coefficient' in p for p in records[i]['paths'])
 save(EXP/'audit/reused_development_rays.json',dict(source=str(rtroot),files={p.name:sha(p) for p in sorted(rtroot.glob('dev_*_rt.json'))},raw_rays_with_complex_coefficients=True,query_ids=ids,not_reusing_csi_or_labels=True))
 # Independent phase construction checks explicit UCA offsets and pilot frequencies.
 z=np.array([[-95,1.73e-6,.4,.9,-1.,2.2]]);alpha=np.array([.2+.3j]);h=channel_from_paths(z,alpha,cfg)
 nf=128;n=8;lam=cfg['propagation']['speed_of_light_mps']/cfg['propagation']['carrier_hz'];radius=lam/(4*np.sin(np.pi/n));phi=2*np.pi*np.arange(n)/n;pos=radius*np.c_[np.cos(phi),np.sin(phi),np.zeros(n)]
 def unit(az,ze):return np.array([np.sin(ze)*np.cos(az),np.sin(ze)*np.sin(az),np.cos(ze)])
 ar=np.exp(2j*np.pi/lam*(pos@unit(z[0,2],z[0,3])))/np.sqrt(n);at=np.exp(-2j*np.pi/lam*(pos@unit(z[0,4],z[0,5])))/np.sqrt(n)
 explicit=alpha[0]*np.exp(-2j*np.pi*(np.arange(nf)*8*60000)*z[0,1])[:,None,None]*ar[None,:,None]*at.conj()[None,None,:]
 phase_error=float(abs(h-explicit).max());assert phase_error<1e-12
 np.testing.assert_allclose(joint_atom(z[0,1:],cfg,128)*alpha[0],explicit.swapaxes(-1,-2).reshape(-1),rtol=1e-12,atol=1e-12)
 assert cfg['propagation']['tau_max_s']<1/cfg['sounding']['pilot_spacing_hz'] and cfg['sounding']['cp_s']>=cfg['propagation']['tau_max_s']
 torch.cuda.reset_peak_memory_stats();est=RFEstimator(cfg);cal=est.calibrate(EXP/'configs/calibration.json');cfg['estimator']['rank_threshold_value']=cal['normalized_rank_threshold'];save(EXP/'configs/effective_config.json',cfg)
 estimators={'historical':RFEstimator(oldcfg,json.loads((old/'historical_configs/calibration.json').read_text())),'revised':est};rows=[];archives={};checks=[];fails=[]
 for protocol,c in [('historical',oldcfg),('revised',cfg)]:
  e=estimators[protocol];signals=[];physical=[]
  for qi in ids:
   h,phys=synth(records[qi],c);physical.append(phys)
   for noise in range(4):
    signal,pilot=simulate_received(h,(2026102910+noise,0,int(qi)),c,noiseless=noise==0);signals.append(signal)
  labels=[];quality=[];elapsed=[]
  for first in range(0,256,8):
   tic=time.perf_counter()
   try:zz,qq=e.estimate_many(np.stack(signals[first:first+8]),pilot)
   except Exception as exc:
    fails.append(dict(protocol=protocol,first=first,batch_error=repr(exc)));zz=[];qq=[]
    for s in signals[first:first+8]:
     try:a,b=e.estimate(s,pilot)
     except Exception as error:a=np.empty((0,6));b=dict(valid=False,reason=repr(error))
     zz.append(a);qq.append(b)
   torch.cuda.synchronize();dt=time.perf_counter()-tic;labels.extend(fold(z) for z in zz);quality.extend(qq);elapsed.extend([dt/len(zz)]*len(zz))
  for j,(z,qual) in enumerate(zip(labels,quality)):
   if not qual['valid']:fails.append(dict(protocol=protocol,frame=j,quality=qual))
   qi=int(ids[j//4]);noise=j%4;reference=labels[(j//4)*4];physical_sq=squared_gospa(z,physical[j//4],c)[0];sq=squared_gospa(z,reference,c)[0]
   rows.append(dict(protocol=protocol,query_id=qi,noise_realization=noise,noiseless=noise==0,n_paths=len(z),weak_paths=int(np.sum(z[:,0]<-105)),valid=qual['valid'],squared_gospa_to_own_noiseless=sq,squared_gospa_to_physical=physical_sq,extraction_seconds=elapsed[j]))
  # Replay all 64 noiseless and one noisy frame per position in scalar mode.
  for j in [k for i in range(64) for k in [4*i,4*i+1]]:
   scalar,squal=e.estimate(signals[j],pilot);scalar=fold(scalar);cost=squared_gospa(scalar,labels[j],c)[0]
   checks.append(dict(protocol=protocol,query_id=int(ids[j//4]),noise_realization=j%4,scalar_valid=squal['valid'],batch_valid=quality[j]['valid'],scalar_K=len(scalar),batch_K=len(labels[j]),difference_squared_gospa=cost))
  zz=np.zeros((256,3,6));kk=np.array([len(z) for z in labels])
  for j,z in enumerate(labels):zz[j,:len(z)]=z
  np.savez_compressed(EXP/f'cache/frontend_pilot_{protocol}.npz',labels=zz,n_paths=kk,query_ids=np.repeat(ids,4),noise=np.tile(np.arange(4),64))
  print('FRONTEND_PILOT',protocol,'done',flush=True)
 csvout(EXP/'results/frontend_pilot_frames.csv',rows);csvout(EXP/'results/frontend_batch_scalar.csv',checks)
 passed=not fails and all(x['scalar_valid'] and x['difference_squared_gospa']<=1e-4 for x in checks)
 summary=dict(passed=passed,created_at=stamp(),frequency_phase_error=phase_error,calibration_trials=cal['trials'],calibration_threshold=cal['normalized_rank_threshold'],calibration_heldout_false_alarm=cal['heldout_false_alarm_rate'],main_extractions=512,additional_scalar_extractions=len(checks),new_rt_traces=0,scalar_max_squared_gospa=max(x['difference_squared_gospa'] for x in checks),failures=fails,device=torch.cuda.get_device_name(),torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,wall_seconds=time.perf_counter()-start,protocols={})
 for name in estimators:
  rr=[x for x in rows if x['protocol']==name];noisy=[x for x in rr if not x['noiseless']]
  summary['protocols'][name]=dict(frames=len(rr),path_count_histogram=np.bincount([x['n_paths'] for x in rr],minlength=4),valid_fraction=np.mean([x['valid'] for x in rr]),noisy_rms_gospa=float(np.sqrt(np.mean([x['squared_gospa_to_own_noiseless'] for x in noisy]))),physical_diagnostic_rms_gospa=float(np.sqrt(np.mean([x['squared_gospa_to_physical'] for x in rr]))),extraction_seconds_sum=sum(x['extraction_seconds'] for x in rr),weak_paths=sum(x['weak_paths'] for x in rr))
 save(EXP/'results/frontend_pilot_summary.json',summary)
 text='# E2 前端小验证\n\n'+('通过' if passed else '未通过')+'。所有结果均来自预先固定的64个开发位置；未读取新确认标签。\n\n推荐配置为60 kHz基础网格、comb=8、128个导频，实际间隔480 kHz。它不是连续60 kHz观测，也不是完整NR波形。\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2,default=lambda v:v.tolist())+'\n```\n\n物理射线对照只作独立诊断；训练器不读取复系数。重用的是带完整复系数的物理缓存，不是旧CSI或旧估计标签。负结果和批量/标量差异全部保留。\n'
 (EXP/'results/frontend_pilot_report.md').write_text(text)
 print(json.dumps({k:summary[k] for k in ['passed','scalar_max_squared_gospa','torch_peak_allocated_bytes','wall_seconds']}),flush=True)
 if not passed:raise RuntimeError('E2 gate failed; see retained diagnostics')
if __name__=='__main__':main()
