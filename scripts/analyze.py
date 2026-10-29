"""Derive every table and figure from all frozen paired runs."""
import csv,json
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import EXP,VARIANTS,save,stamp

def csvwrite(path,rows):
 rows=list(rows);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def interval(values):
 a=np.array([x for x in values if x is not None and np.isfinite(x)],float)
 if len(a)==0:return None,None,None,None
 mean=float(a.mean());sd=float(a.std(ddof=1)) if len(a)>1 else 0.;r=float(stats.t.ppf(.975,len(a)-1)*stats.sem(a)) if len(a)>1 else 0.;return mean,sd,mean-r,mean+r

def analyze():
 tasks=json.loads((EXP/'manifests/formal_jobs.json').read_text());assert len(tasks)==140
 freeze=json.loads((EXP/'configs/freeze.json').read_text());rows=[];sensitivity=[];power=[];cost=[]
 keys=['second_associated_recall','gospa','miss','false','recall','precision','K_mae','predicted_empty_fraction','reference_empty_fraction','empty_false_path_rate','rmse_gain_db','rmse_delay_ns','rmse_aoa_az_deg','rmse_aoa_zen_deg','rmse_aod_az_deg','rmse_aod_zen_deg']
 for task in tasks:
  directory=EXP/task['path'];r=json.loads((directory/'evaluation_confirm.json').read_text());s=r['scores'][0]
  s['second_associated_recall']=s['second_matched']/s['second_reference_count'] if s['second_reference_count'] else None
  assert s['threshold']==freeze['thresholds'][task['variant']]
  training=json.loads((directory/'training.json').read_text());batch=np.load(EXP/f'manifests/batches/formal_{task["pair"]}_{task["ratio"]:.2f}.npz')
  row={k:task[k] for k in ['variant','pair','ratio','lr','steps']};row.update(n_train=len(batch['subset']),threshold=s['threshold'],**{k:s[k] for k in keys});rows.append(row)
  for sec in r['scores']:
   if sec['threshold']==.5:sensitivity.append({**{k:row[k] for k in ['variant','pair','ratio']},**{k:sec[k] for k in ['gospa','recall','precision','predicted_empty_fraction']}})
  for i,band in enumerate(['[-110,-105)','[-105,-95)','[-95,infinity)']):power.append(dict(variant=task['variant'],pair=task['pair'],ratio=task['ratio'],band=band,n=s['power_counts'][i],matched=s['power_matched'][i],recall=s['power_matched'][i]/max(1,s['power_counts'][i])))
  cost.append({k:row[k] for k in ['variant','pair','ratio']}|{k:training[k] for k in ['device','parameter_count','amortized_wall_s','peak_memory_bytes','steps_per_member']})
 csvwrite(EXP/'results/per_flow.csv',rows);csvwrite(EXP/'results/threshold_sensitivity.csv',sensitivity);csvwrite(EXP/'results/power_recall.csv',power);csvwrite(EXP/'results/training_costs.csv',cost)
 summary=[]
 for ratio in [.1,.25,.5,1.]:
  for v in VARIANTS:
   rr=[r for r in rows if r['variant']==v and r['ratio']==ratio]
   if not rr:continue
   assert len(rr)==10
   s=dict(variant=v,ratio=ratio,n=10,n_train_mean=float(np.mean([r['n_train'] for r in rr])),n_train_min=min(r['n_train'] for r in rr),n_train_max=max(r['n_train'] for r in rr),threshold=rr[0]['threshold'],lr=rr[0]['lr'],steps=rr[0]['steps'])
   for k in keys:
    s[k+'_valid_runs']=sum(r[k] is not None and np.isfinite(r[k]) for r in rr)
    for suffix,val in zip(['mean','sd','ci_low','ci_high'],interval([r[k] for r in rr])):s[k+'_'+suffix]=val
   summary.append(s)
 csvwrite(EXP/'results/summary.csv',summary)
 contrasts=[];paired=[]
 for treatment,control in freeze['primary_contrasts']:
  a=sorted([r for r in rows if r['variant']==treatment and r['ratio']==1],key=lambda r:r['pair']);b=sorted([r for r in rows if r['variant']==control and r['ratio']==1],key=lambda r:r['pair'])
  delta=np.array([y['gospa']-x['gospa'] for x,y in zip(a,b)]);rel=np.array([100*(y['gospa']-x['gospa'])/y['gospa'] for x,y in zip(a,b)])
  mean,sd,lo,hi=interval(delta);rmean,rsd,rlo,rhi=interval(rel)
  rng=np.random.Generator(np.random.PCG64DXSM(2026102799));boot=delta[rng.integers(0,10,(10000,10))].mean(1)
  contrasts.append(dict(treatment=treatment,control=control,mean_improvement=mean,sd=sd,ci_low=lo,ci_high=hi,p=(float(stats.ttest_1samp(delta,0).pvalue) if np.std(delta)>1e-15 else (1. if np.max(np.abs(delta))<1e-15 else 0.)),mean_relative_improvement_percent=rmean,relative_ci_low=rlo,relative_ci_high=rhi,favorable_pairs=int((delta>0).sum()),bootstrap_ci_low=float(np.quantile(boot,.025)),bootstrap_ci_high=float(np.quantile(boot,.975))))
  paired.extend(dict(treatment=treatment,control=control,pair=r['pair'],absolute_improvement=float(d),relative_improvement_percent=float(rel[i])) for i,(r,d) in enumerate(zip(a,delta)))
 previous=0.
 for rank,i in enumerate(sorted(range(len(contrasts)),key=lambda i:contrasts[i]['p'])):
  previous=max(previous,min(1.,(len(contrasts)-rank)*contrasts[i]['p']));contrasts[i]['p_holm']=previous
 csvwrite(EXP/'results/primary_contrasts.csv',contrasts);csvwrite(EXP/'results/paired_differences.csv',paired)
 ref=np.load(EXP/'cache/reference_confirm.npz');save(EXP/'results/statistical_summary.json',dict(created_at=stamp(),contrasts=contrasts,all_empty_gospa=float(np.sqrt(4.5*ref['n_paths'].mean())),reference_empty_fraction=float((ref['n_paths']==0).mean()),primary_comparisons=4,interpretation='Ten paired observation-noise/optimization repetitions in one fixed city; confirmation queries are not independent experimental replicates.'))
 (EXP/'figures').mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],'font.size':9,'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42})
 colors={'Coord':'#bd4a36','RawMap':'#c58b12','GeoPool':'#6f5fa9','GeoDirect':'#5c8b72','Full':'#176a9f'}
 fig,ax=plt.subplots(figsize=(3.5,2.45))
 for v,marker in [('Coord','s'),('RawMap','^'),('Full','o')]:
  ss=sorted([s for s in summary if s['variant']==v],key=lambda r:r['ratio']);x=[s['n_train_mean'] for s in ss];y=[s['gospa_mean'] for s in ss]
  ax.plot(x,y,marker=marker,ms=4,lw=1.3,color=colors[v],label=v);ax.fill_between(x,[s['gospa_ci_low'] for s in ss],[s['gospa_ci_high'] for s in ss],alpha=.13,color=colors[v],linewidth=0)
 ax.set(xlabel='Training observations',ylabel='RMS GOSPA');ax.grid(alpha=.2,lw=.6);ax.legend(fontsize=8,framealpha=.95);fig.tight_layout(pad=.5)
 for ext in ['pdf','png']:fig.savefig(EXP/f'figures/sample_efficiency.{ext}',dpi=220,bbox_inches='tight',pad_inches=.02)
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(3.5,2.3));ss=[next(s for s in summary if s['variant']==v and s['ratio']==1) for v in VARIANTS]
 ax.bar(np.arange(5),[s['gospa_mean'] for s in ss],yerr=[s['gospa_mean']-s['gospa_ci_low'] for s in ss],color=[colors[v] for v in VARIANTS],capsize=3,width=.65)
 ax.set_xticks(np.arange(5),['Coord','Raw map','Pooled\ngeometry','No\nanchors','Full']);ax.set_ylabel('RMS GOSPA');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout(pad=.5)
 for ext in ['pdf','png']:fig.savefig(EXP/f'figures/feature_attribution.{ext}',dpi=220,bbox_inches='tight',pad_inches=.02)
 plt.close(fig)
 save(EXP/'results/analysis_complete.json',dict(completed_at=stamp(),runs=len(rows),summary_rows=len(summary),primary_comparisons=len(contrasts)))
 print(json.dumps({'full_data':[s for s in summary if s['ratio']==1],'contrasts':contrasts},indent=2,default=lambda x:x.tolist()))
if __name__=='__main__':analyze()
