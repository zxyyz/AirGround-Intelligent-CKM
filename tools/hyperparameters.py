"""Supplementary parameter rationale and sensitivity; frozen models are never tuned."""
import argparse,copy,csv,itertools,json,sys,time
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from paths import EXP,config,save,sha,stamp,VARIANTS
from ckm_sim.sounding import array_response,channel_from_paths,simulate_received
from ckm_sim.metrics import path_squared_distance,squared_gospa

def csvwrite(name,rows):
 with (EXP/'results'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def physical():
 cfg=config();n=cfg['sounding']['nr'];theta=np.pi/2
 def corr(delta):
  ref=array_response(np.array([0.,theta]),n,cfg,True)
  a=array_response(np.stack([np.asarray(delta),np.full_like(np.asarray(delta),theta)],-1),n,cfg,True)
  return abs(np.sum(ref.conj()*a,axis=-1))**2
 grid=np.linspace(0,np.pi,18001);v=corr(grid)-.5;j=np.flatnonzero((v[:-1]>0)&(v[1:]<=0))[0]
 half=np.rad2deg(brentq(lambda d:float(corr(d)-.5),grid[j],grid[j+1],xtol=1e-14))
 df=cfg['sounding']['delta_f_hz'];nf=cfg['sounding']['n_frequencies'];q=cfg['estimator']['frequency_window_q']
 result=dict(n_elements=n,nominal_bandwidth_hz=nf*df,endpoint_span_hz=(nf-1)*df,nominal_delay_ns=1e9/(nf*df),smoothed_delay_ns=1e9/(q*df),azimuth_half_power_half_width_deg=half,azimuth_HPBW_full_width_deg=2*half,notes='Actual UCA steering at azimuth0/zenith90. Beamwidth and inverse bandwidth are physical reference scales, not MUSIC RMSE, a universal 3D resolution or a hard two-path limit. Gain5dB is an engineering weighting.')
 save(EXP/'results/hyperparameter_physical_reference.json',result)
 rows=[('S0',5,25,15,3),('Gain_strict',3,25,15,3),('Gain_loose',8,25,15,3),('Delay_10',5,10,15,3),('Delay_15',5,15,15,3),('Delay_40',5,40,15,3),('Angle_10',5,25,10,3),('Angle_25',5,25,25,3),('Joint_strict',3,10,10,3),('Joint_relaxed',8,40,25,3),('S_phys',5,1e9/(nf*df),2*half,3),('S_phys_Q',5,1e9/(q*df),2*half,3),('S_phys_half',5,1e9/(nf*df),half,3),('Cutoff_2',5,25,15,2),('Cutoff_4',5,25,15,4)]
 protocol=dict(created_at=stamp(),status='frozen_before_new_confirmation',requested_during_formal_training=False,role='Descriptive sensitivity, separate from prespecified primary confirmation; no parameter selection or retraining.',scales=[dict(zip(['id','gain_db','delay_ns','angle_deg','cutoff'],r)) for r in rows],comparison_scope='50 full-data ensembles; fixed weights, thresholds, predictions and medoid; recompute assignment under each scale',path_caps=[1,2,3,4,6],front_end='Use identical archived Sionna rays and noiseless signals with retained target cap6; verify its first3 reproduce the original reference.',prohibitions=['No new training scale selection','No replacement of the primary 3-path target','No pooling historical and current outcomes'],source_sha256=sha(Path(__file__)))
 target=EXP/'configs/supplementary_protocol.json'
 if target.exists():
  prior=json.loads(target.read_text());assert prior['scales']==protocol['scales']
 else:save(target,protocol)


def records():
 output=[]
 for p in sorted((EXP/'cache/reference_parts').glob('confirm_*_rt.json')):output.extend(json.loads(p.read_text()))
 assert len(output)==4096;return output

def coverage():
 c=config();rt=records();rows=[]
 for i,r in enumerate(rt):
  legal=[x for x in r['paths'] if c['propagation']['tau_min_s']<=x['z'][1]<=c['propagation']['tau_max_s'] and x['z'][0]>=-110]
  signal=[r['paths'][j] for j in r['signal_indices']];power=np.array([10**(x['z'][0]/10) for x in legal]);ps=np.array([10**(x['z'][0]/10) for x in signal]);power=np.sort(power)[::-1];ps=np.sort(ps)[::-1]
  row=dict(query=i,physical_above_floor=len(power),signal_components=len(signal),signal_above_floor=sum(x['z'][0]>=-110 for x in signal),uncapped_power_sum=float(power.sum()),signal_power_sum=float(ps.sum()))
  for k in [1,2,3,4,6]:row[f'top{k}_above_floor_power']=float(power[:k].sum());row[f'top{k}_signal_power']=float(ps[:k].sum())
  rows.append(row)
 csvwrite('path_cap_per_query.csv',rows);summary=[]
 for denominator,column in [('uncapped_above_-110dB','physical_above_floor'),('synthesized_signal_up_to6','signal_components')]:
  totalkey='uncapped_power_sum' if column=='physical_above_floor' else 'signal_power_sum';suffix='above_floor_power' if column=='physical_above_floor' else 'signal_power'
  active=[r for r in rows if r[column]>0]
  for k in [1,2,3,4,6]:
   fractions=np.array([r[f'top{k}_{suffix}']/r[totalkey] for r in active]);truncated=sum(r[column]>k for r in rows)
   summary.append(dict(population=denominator,cap=k,queries=4096,nonempty_queries=len(active),queries_truncated=truncated,truncated_fraction_all=truncated/4096,truncated_fraction_nonempty=truncated/len(active),mean_noncoherent_power_fraction=float(fractions.mean()),p05_noncoherent_power_fraction=float(np.quantile(fractions,.05)),aggregate_noncoherent_power_fraction=sum(r[f'top{k}_{suffix}'] for r in rows)/sum(r[totalkey] for r in rows)))
 csvwrite('path_cap_coverage.csv',summary)
 geo=json.loads((EXP/'results/geometry_confirm.json').read_text());lost=[max(0,r['valid']-r['valid_retained']) for r in geo]
 save(EXP/'results/path_cap_physical_diagnostic.json',dict(created_at=stamp(),scope='Recorded Sionna rays, fixed delay support and -110dB floor; power sums are noncoherent, not the coherent channel-energy norm.',rows=summary,geometry_capacity=dict(capacity=64,maximum_valid_candidates=max(r['valid'] for r in geo),queries_with_valid_candidate_truncation=sum(x>0 for x in lost),total_valid_candidates_lost=sum(lost))))
 print('PHYSICAL_PATH_CAP_COVERAGE_READY',flush=True)


def front_end():
 import torch
 from collector import Receiver
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 c=config();c['ckm_target']['max_paths']=6;c['runtime']['device']='cuda:0';receiver=Receiver(c);rt=records();out=EXP/'cache/reference_uncapped_diagnostic.npz';labels=[];quality=[];start=time.time()
 if out.exists():a=np.load(out);z=a['reference'];k=a['n_paths']
 else:
  for first in range(0,len(rt),8):
   signals=[]
   for i,r in enumerate(rt[first:first+8],first):
    raw=[r['paths'][j] for j in r['signal_indices']];z=np.array([x['z'] for x in raw]).reshape(-1,6);alpha=np.array([complex(*x['coefficient']) for x in raw])*np.exp(-2j*np.pi*c['propagation']['carrier_hz']*z[:,1]);h=channel_from_paths(z,alpha,c);signal,p=simulate_received(h,(2026102702,0,i),c,noiseless=True);signals.append(signal)
   zz,qq=receiver.estimate(np.stack(signals),p);assert all(x['valid'] for x in qq);labels.extend(zz);quality.extend(qq)
  k=np.array([len(x) for x in labels],np.int32);z=np.zeros((4096,6,6),float)
  for i,x in enumerate(labels):z[i,:len(x)]=x
  np.savez_compressed(out,reference=z,n_paths=k)
 original=np.load(EXP/'cache/reference_confirm.npz');assert np.array_equal(np.minimum(k,3),original['n_paths'])
 # Same GPU/library replay should be nearly identical; record exact numerical residual.
 active=np.arange(3)[None,:]<original['n_paths'][:,None];err=float(np.max(np.abs(z[:,:3][active]-original['reference'][active])))
 np.testing.assert_allclose(z[:,:3][active],original['reference'][active],rtol=1e-6,atol=2e-6)
 rows=[]
 for cap in [1,2,3,4,6]:
  mask=np.arange(6)[None,:]<k[:,None];p=np.where(mask,10**(z[:,:,0]/10),0);total=p.sum(1);nonempty=total>0;ret=p[:,:cap].sum(1)
  rows.append(dict(cap=cap,retained_count=int(np.minimum(k,cap).sum()),total_components=int(k.sum()),queries_truncated=int((k>cap).sum()),queries_truncated_fraction=float((k>cap).mean()),mean_noncoherent_power_fraction=float((ret[nonempty]/total[nonempty]).mean()),aggregate_noncoherent_power_fraction=float(ret.sum()/total.sum())))
 csvwrite('path_cap_frontend.csv',rows)
 save(EXP/'results/path_cap_frontend_diagnostic.json',dict(created_at=stamp(),reference_sha256=sha(out),counts=np.bincount(k,minlength=7).tolist(),original_top3_max_absolute_difference=err,original_top3_cardinality_equal=True,rows=rows,wall_s=time.time()-start,interpretation='Retained MUSIC-LS components from the same signal/front end, with cap6 only after gain fitting; this is a diagnostic, not a retrained Kmax6 learner.'))
 print('UNCAPPED_FRONTEND_DIAGNOSTIC',np.bincount(k,minlength=7).tolist(),err,flush=True)


def matched(pred,k,ref,rk,scale):
 c=copy.deepcopy(config());m=c['metric'];m.update(gain_scale_db=scale['gain_db'],delay_scale_s=scale['delay_ns']*1e-9,aoa_direction_scale_deg=scale['angle_deg'],aod_direction_scale_deg=scale['angle_deg'],gospa_c=scale['cutoff'])
 d=path_squared_distance(np.asarray(pred,float)[:,:,None],np.asarray(ref,float)[:,None],c);penalty=scale['cutoff']**2/2;best=penalty*(k+rk);mapping=np.full((len(k),3),-1,np.int8)
 for n in range(1,4):
  for ai in itertools.combinations(range(3),n):
   for bj in itertools.permutations(range(3),n):
    cost=penalty*(k+rk-2*n)+d[:,ai,bj].sum(-1);good=(k>max(ai))&(rk>max(bj))&(cost<best-1e-12);best[good]=cost[good];mapping[good]=-1
    for i,j in zip(ai,bj):mapping[good,i]=j
 return best,mapping,c


def rescore():
 protocol=json.loads((EXP/'configs/supplementary_protocol.json').read_text());tasks=[t for t in json.loads((EXP/'manifests/formal_jobs.json').read_text()) if t['ratio']==1];assert len(tasks)==50
 reference=np.load(EXP/'cache/reference_confirm.npz');r=reference['reference'];rk=reference['n_paths'];rows=[];max_error=0;scalar_replays=0
 for t in tasks:
  directory=EXP/t['path'];e=json.loads((directory/'evaluation_confirm.json').read_text())['scores'][0];a=np.load(directory/f'predictions_confirm_{e["threshold"]:.2f}.npz');pred=a['prediction'];k=a['predicted_K']
  for scale in protocol['scales']:
   sq,mapping,cfg=matched(pred,k,r,rk,scale)
   if scale['id']=='S0':np.testing.assert_allclose(sq,a['squared_gospa'],rtol=0,atol=1e-11);np.testing.assert_array_equal(mapping,a['matching'])
   # Scalar checks span every scale and all cardinalities for each paired run.
   for count in range(4):
    ids=np.flatnonzero(rk==count)
    for i in ids[:2]:
     scalar,pairs,_,_=squared_gospa(pred[i,:k[i]],r[i,:rk[i]],cfg);error=abs(float(sq[i])-scalar);max_error=max(max_error,error);assert error<1e-10;assert list(pairs)==[(a,int(b)) for a,b in enumerate(mapping[i]) if b>=0];scalar_replays+=1
   nm=(mapping>=0).sum(-1);rows.append(dict(scale=scale['id'],variant=t['variant'],pair=t['pair'],gospa=float(np.sqrt(sq.mean())),recall=float(nm.sum()/rk.sum()),mean_miss=float((rk-nm).mean()),mean_false=float((k-nm).mean()),matching_changed_fraction=float(np.any(mapping!=a['matching'],axis=1).mean())))
 csvwrite('scale_sensitivity_per_flow.csv',rows);comparisons=[]
 for scale in protocol['scales']:
  for treatment,control in json.loads((EXP/'configs/freeze.json').read_text())['primary_contrasts']:
   aa=sorted([r for r in rows if r['scale']==scale['id'] and r['variant']==treatment],key=lambda r:r['pair']);bb=sorted([r for r in rows if r['scale']==scale['id'] and r['variant']==control],key=lambda r:r['pair']);delta=np.array([y['gospa']-x['gospa'] for x,y in zip(aa,bb)]);relative=np.array([100*(y['gospa']-x['gospa'])/y['gospa'] for x,y in zip(aa,bb)]);half=float(stats.t.ppf(.975,9)*stats.sem(delta));rh=float(stats.t.ppf(.975,9)*stats.sem(relative))
   comparisons.append(dict(scale=scale['id'],treatment=treatment,control=control,mean_improvement=float(delta.mean()),ci_low=float(delta.mean()-half),ci_high=float(delta.mean()+half),mean_relative_improvement_percent=float(relative.mean()),relative_ci_low=float(relative.mean()-rh),relative_ci_high=float(relative.mean()+rh),favorable_pairs=int((delta>0).sum())))
 csvwrite('scale_sensitivity_contrasts.csv',comparisons)
 save(EXP/'results/scale_sensitivity_validation.json',dict(passed=True,ensembles=50,scales=15,query_evaluations=50*15*4096,scalar_replays=scalar_replays,maximum_scalar_error=max_error,baseline_reproduced=True,fixed_predictions=True,medoid_reselected=False,retrained=False,created_at=stamp()))
 print('SCALE_SENSITIVITY_VALIDATED',max_error,scalar_replays,flush=True)


def report():
 physical=json.loads((EXP/'results/hyperparameter_physical_reference.json').read_text());coverage=json.loads((EXP/'results/path_cap_physical_diagnostic.json').read_text());front=json.loads((EXP/'results/path_cap_frontend_diagnostic.json').read_text())
 with (EXP/'results/scale_sensitivity_contrasts.csv').open() as f:contrasts=list(csv.DictReader(f))
 text=['# 核心超参数依据与补充验证','\n本补充按2026-10-29修订计划在新确认参考生成前冻结，属于描述性分析，不增加四项主要确认检验。保持主训练、阈值、三路径目标与medoid输出不变，固定15组尺度全部报告；没有根据补充结果重新选择超参数。','\n## 路径数与候选容量','\n信号合成最多6条，MUSIC拟合最多6条，CKM目标最多3条，三者是不同限制。Kmax=3定义紧凑的主导路径集并控制多目标匹配/学习复杂度，不表示物理上最多只有三条路径。以下功率比例是非相干路径功率和，不能等同于含相位干涉的信道能量比例。','\n| 保留条数 | MUSIC参考中被截断查询% | MUSIC路径功率和平均保留% | MUSIC路径功率和总体保留% |\n|---:|---:|---:|---:|']
 for r in front['rows']:text.append(f'| {r["cap"]} | {100*r["queries_truncated_fraction"]:.3f} | {100*r["mean_noncoherent_power_fraction"]:.4f} | {100*r["aggregate_noncoherent_power_fraction"]:.4f} |')
 g=coverage['geometry_capacity'];text.extend([f'\n在新4096查询中，cap6参考的前3条与正式参考路径数完全一致，最大参数绝对差{front["original_top3_max_absolute_difference"]:.3g}。公开几何有效候选数最大为{g["maximum_valid_candidates"]}，64槽容量截掉有效候选的查询数为{g["queries_with_valid_candidate_truncation"]}。物理射线的未截断计数与功率覆盖另见path_cap_coverage.csv；不能只用已经截到3条的正式标签自证3条足够。','\n这些覆盖结果解释本场景的紧凑目标取舍，不证明3是所有城市的最优路径数，也不替代Kmax4/6下重新训练的实验。','\n## 去量纲尺度','\n5dB、25ns、15°是固定工程误差尺度，分别控制增益、时延、双端方向在联合距离中的相对权重，并非标准规定或由分辨率唯一推导。增益5dB不能由带宽/阵列孔径推出。',f'\n本配置128×480kHz的名义孔径给出1/B={physical["nominal_delay_ns"]:.3f}ns，64点平滑孔径给出{physical["smoothed_delay_ns"]:.3f}ns。实际8元水平UCA在指定水平入射方向的方位半功率半宽为{physical["azimuth_half_power_half_width_deg"]:.6f}°，全宽为{physical["azimuth_HPBW_full_width_deg"]:.6f}°。15°接近该特定方向半宽，25ns位于两个名义时延参照之间，但这种接近不能包装成严格物理推导。方位宽度依方向变化，水平UCA还存在上下镜像歧义；上述参照不是MUSIC误差或统一三维分辨率。','\n重新计算每个尺度的最优匹配，不能仅将旧总误差乘比例。比较固定五组全数据模型及各10流，15尺度包含单项/联合严格宽松、带宽及实际阵列参照、截止2/4；不重选medoid，也不更换训练损失。这里验证的是固定模型评价结论的尺度敏感性，不证明改变训练权重后训练结果等价。','\n| 对比 | 15尺度平均配对改善范围% | 正改善尺度数/15 | 95%区间全正尺度数/15 |\n|---|---:|---:|---:|'])
 for treatment,control in json.loads((EXP/'configs/freeze.json').read_text())['primary_contrasts']:
  rr=[r for r in contrasts if r['treatment']==treatment and r['control']==control];values=[float(r['mean_relative_improvement_percent']) for r in rr];text.append(f'| {treatment} / {control} | {min(values):.3f}–{max(values):.3f} | {sum(float(r["mean_improvement"])>0 for r in rr)} | {sum(float(r["ci_low"])>0 for r in rr)} |')
 text.extend(['\n区间为同一10流的配对描述，不把15个相关尺度当作15次独立确认；没有跨尺度显著性优选，也不跨不同尺度比较绝对GOSPA。完整60条对比及750条逐流结果均纳入本地发布候选；线上发布另行授权。','\n## 其他主要参数的解释','\n| 参数 | 选择依据与本轮验证边界 |\n|---|---|\n| GOSPA c=3, p=2, alpha=2 | c定义联合归一化距离的匹配容差；单个漏/虚径成本为4.5。附带c=2/4敏感性，不声称普适最优。 |\n| 坐标240/240/120m归一化 | 来自公开场景几何范围，不使用确认集统计。 |\n| 输出增益中心-100dB、尺度20dB | 网络输出的固定仿射参数化，与指标5dB的权重是两回事。 |\n| 延迟上限1.8μs | 原保留时延范围，与2μs循环前缀一致；保持原物理前端定义。 |\n| 残差50ns、方向分量0.2 | 沿用原学习器的有界修正范围；锚定消融验证整体作用，不单独宣称两个范围各自最优。 |\n| 近重复5ns、5° | 原推理/前端的数值近重复阈值，不是物理两径分辨率。 |\n| 三成员medoid | 沿用原稳健集合输出；本轮没有把成员数当作新增优化变量。 |\n| 学习率、存在阈值、更新步数 | 按3条开发流、固定网格、组间共享规则与统一延长规则选择，均在确认参考生成前冻结。 |'])
 (EXP/'results/HYPERPARAMETER_REPORT_CN.md').write_text('\n'.join(text)+'\n')
 save(EXP/'results/hyperparameter_supplement_complete.json',dict(completed_at=stamp(),protocol_sha256=sha(EXP/'configs/supplementary_protocol.json'),source_sha256=sha(Path(__file__)),main_freeze_unchanged=True))
 print('HYPERPARAMETER_REPORT_READY',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['design','physical','front-end','rescore','report','after']);a=p.parse_args()
 if a.action=='design':physical();coverage()
 elif a.action=='after':
  while not (EXP/'results/postprocessing_complete.json').exists():time.sleep(30)
  front_end();rescore();report()
 else:{'physical':coverage,'front-end':front_end,'rescore':rescore,'report':report}[a.action]()
