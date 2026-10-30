"""Read-only merge of initial 140 and added 60 groups; all outputs stay here."""
from common import *
import csv,gzip,math
import numpy as np
from scipy import stats
from analyze import interval
V=['Coord','RawMap','GeoPool','GeoDirect','Full']
N={v:('Proposed' if v=='Full' else v) for v in V}
KEYS=['second_associated_recall','gospa','miss','false','recall','precision','K_mae','predicted_empty_fraction','reference_empty_fraction','empty_false_path_rate','rmse_gain_db','rmse_delay_ns','rmse_aoa_az_deg','rmse_aoa_zen_deg','rmse_aod_az_deg','rmse_aod_zen_deg']

def csvout(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as out:
  w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 assert read(EXT/'results/validation_complete.json')['passed']
 original=read(BASE/'manifests/formal_jobs.json');added=tasks()
 combined=[{**t,'origin':'original_140','source_root':str(BASE)} for t in original]+[{**t,'origin':'extension_60','source_root':str(EXT)} for t in added]
 assert len(combined)==len({(t['variant'],t['pair'],t['ratio']) for t in combined})==200
 save(EXT/'manifests/combined_tasks.json',combined)
 rows=[];sources={};freeze=read(BASE/'configs/freeze.json')
 for t in combined:
  root=Path(t['source_root']);directory=root/t['path'];ev=read(directory/'evaluation_confirm.json');s=ev['scores'][0];tr=read(directory/'training.json')
  assert s['threshold']==freeze['thresholds'][t['variant']]
  s['second_associated_recall']=s['second_matched']/s['second_reference_count'] if s['second_reference_count'] else None
  batch=np.load(BASE/f'manifests/batches/formal_{t["pair"]}_{t["ratio"]:.2f}.npz')
  row={k:t[k] for k in ['variant','pair','ratio','lr','steps']};row.update(n_train=len(batch['subset']),threshold=s['threshold'],**{k:s[k] for k in KEYS});rows.append(row)
  for name in ['evaluation_confirm.json','training.json']:sources[str(directory/name)]=sha(directory/name)
 key=lambda r:(r['variant'],int(r['pair']),float(r['ratio']))
 with (BASE/'results/per_flow.csv').open() as f:old=list(csv.DictReader(f))
 lookup={key(r):r for r in rows}
 checks=0
 for r in old:
  now=lookup[key(r)]
  for k,v in r.items():
   assert (str(now[k])==v if k=='variant' else now[k] is None if v=='' else float(now[k])==float(v)),(key(r),k,now[k],v)
   checks+=1
 rows.sort(key=lambda r:([.1,.25,.5,1.].index(r['ratio']),V.index(r['variant']),r['pair']))
 csvout(EXT/'results/per_flow_200.csv',rows)
 summary=[]
 for ratio in [.1,.25,.5,1.]:
  for v in V:
   rr=[r for r in rows if r['variant']==v and r['ratio']==ratio];assert len(rr)==10
   s=dict(variant=v,display_name=N[v],ratio=ratio,n=10,n_train_mean=float(np.mean([r['n_train'] for r in rr])),n_train_min=min(r['n_train'] for r in rr),n_train_max=max(r['n_train'] for r in rr),threshold=rr[0]['threshold'],lr=rr[0]['lr'],steps=56800)
   for k in KEYS:
    s[k+'_valid_runs']=sum(r[k] is not None and np.isfinite(r[k]) for r in rr)
    for suffix,value in zip(['mean','sd','ci_low','ci_high'],interval([r[k] for r in rr])):s[k+'_'+suffix]=value
   summary.append(s)
 assert len(summary)==20
 csvout(EXT/'results/summary_20_cells.csv',summary)
 with (BASE/'results/summary.csv').open() as f:
  for oldrow in csv.DictReader(f):
   r=next(s for s in summary if s['variant']==oldrow['variant'] and s['ratio']==float(oldrow['ratio']))
   for k,v in oldrow.items():assert (str(r[k])==v if k=='variant' else r[k] is None if v=='' else float(r[k])==float(v)),(k,r[k],v)
 comparisons=[];paired=[]
 for ratio in [.1,.25,.5,1.]:
  for treatment,control in freeze['primary_contrasts']:
   a=sorted([r for r in rows if r['variant']==treatment and r['ratio']==ratio],key=lambda r:r['pair']);b=sorted([r for r in rows if r['variant']==control and r['ratio']==ratio],key=lambda r:r['pair'])
   delta=np.array([y['gospa']-x['gospa'] for x,y in zip(a,b)]);rel=np.array([100*(y['gospa']-x['gospa'])/y['gospa'] for x,y in zip(a,b)])
   mean,sd,lo,hi=interval(delta);rm,rs,rl,rh=interval(rel)
   comparisons.append(dict(treatment=treatment,control=control,ratio=ratio,absolute_improvement=mean,absolute_sd=sd,ci_low=lo,ci_high=hi,relative_improvement_percent=rm,relative_ci_low=rl,relative_ci_high=rh,favorable_pairs=int((delta>0).sum()),n=10,interpretation='descriptive extension; original full-data primary tests remain separately unchanged'))
   paired.extend(dict(treatment=treatment,control=control,ratio=ratio,pair=r['pair'],absolute_improvement=float(d),relative_improvement_percent=float(rel[i])) for i,(r,d) in enumerate(zip(a,delta)))
 csvout(EXT/'results/descriptive_contrasts_by_ratio.csv',comparisons);csvout(EXT/'results/paired_differences_by_ratio.csv',paired)
 with gzip.open(EXT/'results/per_query_200.csv.gz','wt',newline='') as out:
  with gzip.open(BASE/'results/per_query.csv.gz','rt') as src:
   header=next(src);out.write(header)
   for line in src:out.write(line)
  for t in added:
   with gzip.open(EXT/f'results/per_query/{t["variant"]}_{t["pair"]}_{t["ratio"]:.2f}.csv.gz','rt') as src:
    assert next(src).strip()==header.strip()
    for line in src:out.write(line)
 with gzip.open(EXT/'results/per_query_200.csv.gz','rt') as stream:count=sum(1 for _ in stream)-1
 assert count==200*4096
 save(EXT/'results/aggregation_validation.json',dict(passed=True,groups=200,members=600,summary_cells=20,development_groups=45,development_members=135,original_groups=140,additional_groups=60,original_per_flow_fields_identical=checks,original_summary_cells_identical=14,queries_exported=count,created_at=stamp()))
 save(EXT/'results/data_sources.json',dict(created_at=stamp(),baseline_freeze_sha256=sha(BASE/'configs/freeze.json'),original_per_flow_sha256=sha(BASE/'results/per_flow.csv'),original_summary_sha256=sha(BASE/'results/summary.csv'),sources=sources,outputs={name:sha(EXT/'results'/name) for name in ['summary_20_cells.csv','per_flow_200.csv','descriptive_contrasts_by_ratio.csv','paired_differences_by_ratio.csv','per_query_200.csv.gz']}))
 print('AGGREGATED',len(rows),len(summary),count,flush=True)
if __name__=='__main__':main()
