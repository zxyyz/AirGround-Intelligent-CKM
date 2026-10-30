"""E1/E4 physical interpretation from frozen medoid predictions; no tuning."""
import argparse,csv,json,itertools,copy
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import EXP,save,stamp
from ckm_sim.geometry import directions
from ckm_sim.metrics import squared_gospa,spherical_angle,path_squared_distance
NAMES={'Coord':'Coordinates only','RawMap':'Direct map learning','GeoPool':'Pooled geometry features','GeoDirect':'Candidate model without anchors','Full':'Proposed'}
COLORS={'Coord':'#b34a3d','RawMap':'#bf851c','GeoPool':'#7d65a5','GeoDirect':'#418c74','Full':'#176b9c'}
COMP=['gain','delay','aoa','aod','miss','false']
def csvout(path,rows):
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def matching(a,ak,b,bk,cfg):
 d=path_squared_distance(a[:,:,None],b[:,None,:],cfg);pen=cfg['metric']['gospa_c']**2/2
 best=pen*(ak+bk);mp=np.full((len(a),3),-1,int)
 for n in range(1,4):
  for ai in itertools.combinations(range(3),n):
   for bj in itertools.permutations(range(3),n):
    val=pen*(ak+bk-2*n)+d[:,ai,bj].sum(-1);ok=(ak>max(ai))&(bk>max(bj))&(val<best-1e-12)
    best[ok]=val[ok];mp[ok]=-1
    for i,j in zip(ai,bj):mp[ok,i]=j
 return best,mp

def examples(cfg,out):
 z=np.array([[-95.,100e-9,0.,np.pi/2,0.,np.pi/2]]);z2=z.copy();z2[:,1]+=300e-9;z3=z.copy();z3[:,1]+=600e-9
 cases=[('identical',z,z,0),('both_empty',z[:0],z[:0],0),('order_swap',np.r_[z,z2],np.r_[z2,z],0)]
 for title,axis,delta in [('gain_5dB',0,5),('delay_25ns',1,25e-9),('aoa_15deg',2,np.pi/12),('aod_15deg',4,np.pi/12)]:
  p=z.copy();p[:,axis]+=delta;cases.append((title,p,z,1))
 cases.extend([('one_missed',z[:0],z,4.5),('one_false',z,z[:0],4.5),('all_three_missed',z[:0],np.r_[z,z2,z3],13.5)])
 p=z.copy();p[:,0]+=15;cases.append(('match_cost9_equals_miss_plus_false',p,z,9))
 rows=[]
 for title,a,b,want in cases:
  sq,mp,mi,fa=squared_gospa(a,b,cfg);assert abs(sq-want)<1e-10
  rows.append(dict(case=title,prediction=a.tolist(),reference=b.tolist(),squared_gospa=sq,gospa=np.sqrt(sq),matching=mp,miss=mi,false=fa))
 save(out/'metric_examples.json',dict(examples=rows,units='dimensionless weighted set distance; not an application acceptance threshold',mse_mae='Applicable after alignment for equal-length vectors; additional correspondence and cardinality treatment needed for unordered variable-size sets.'))

def run(source,prototype):
 cfg=json.loads((source/'configs/effective_config.json').read_text());out=EXP/'results'/('prototype' if prototype else 'final');fig=EXP/'figures'/('prototype' if prototype else 'final');out.mkdir(exist_ok=True);fig.mkdir(exist_ok=True)
 examples(cfg,out)
 ref=np.load(source/'cache/reference_confirm.npz');r=ref['reference'];rk=ref['n_paths'];tasks=json.loads((source/'manifests/formal_jobs.json').read_text());assert len(tasks)==140
 rows=[];components=[];strata=[];sens=[];scalar=0;maxerr=0.;squared_err=0.;example_archive={};all_errors={};curves=[]
 scales=json.loads((source/'configs/supplementary_protocol.json').read_text())['scales']
 reused_validation=None
 if not prototype:
  reused_validation=json.loads((source/'results/final_validation.json').read_text())
  assert reused_validation['passed'] and reused_validation['ensemble_runs']==140 and reused_validation['scalar_replays']>=10000
  scalar=reused_validation['scalar_replays'];squared_err=reused_validation['maximum_absolute_error']
 for t in tasks:
  di=source/t['path'];ev=json.loads((di/'evaluation_confirm.json').read_text());e=ev['scores'][0];a=np.load(di/f'predictions_confirm_{e["threshold"]:.2f}.npz');p=a['prediction'];pk=a['predicted_K'];mp=a['matching'];sq=a['squared_gospa'];nm=(mp>=0).sum(1);qidx,pi=np.where(mp>=0);ri=mp[qidx,pi]
  diff=p[qidx,pi]-r[qidx,ri];diff[:,[2,4]]=(diff[:,[2,4]]+np.pi)%(2*np.pi)-np.pi
  err=np.c_[abs(diff[:,0]),abs(diff[:,1])*1e9,*[np.rad2deg(spherical_angle(directions(p[qidx,pi,j:j+2]),directions(r[qidx,ri,j:j+2]))) for j in [2,4]]]
  cc=np.zeros((len(p),6));m=cfg['metric'];scale=np.array([m['gain_scale_db'],m['delay_scale_s']*1e9,m['aoa_direction_scale_deg'],m['aod_direction_scale_deg']])
  for j in range(4):np.add.at(cc[:,j],qidx,(err[:,j]/scale[j])**2)
  cc[:,4]=(rk-nm)*m['gospa_c']**2/2;cc[:,5]=(pk-nm)*m['gospa_c']**2/2
  (out/'query_components').mkdir(exist_ok=True)
  np.savez_compressed(out/f'query_components/{t["variant"]}_{t["pair"]}_{t["ratio"]:.2f}.npz',query_id=np.arange(len(p)),components=cc,squared_gospa=sq,matching=mp,reference_K=rk,predicted_K=pk)
  residual=float(abs(cc.sum(1)-sq).max());maxerr=max(maxerr,residual);assert residual<1e-8
  # Final E3 validation already replays >=10,000 matches; E1 newly checks every component identity.
  if prototype:
   checks=[]
   for k in range(4):
    ids=np.flatnonzero(rk==k);checks.extend(ids[np.linspace(0,len(ids)-1,min(20,len(ids)),dtype=int)])
   for qi in checks:
    val,_,_,_=squared_gospa(p[qi,:pk[qi]],r[qi,:rk[qi]],cfg);squared_err=max(squared_err,abs(val-sq[qi]));assert abs(val-sq[qi])<1e-8;scalar+=1
  base=dict(variant=t['variant'],display_name=NAMES[t['variant']],flow=t['pair'],ratio=t['ratio'],threshold=e['threshold'])
  curves.append(dict(**base,rms_gospa=float(np.sqrt(sq.mean()))))
  components.append(dict(**base,**{k:float(v) for k,v in zip(COMP,cc.mean(0))},rms_gospa=float(np.sqrt(sq.mean())),identity_residual=residual))
  row=dict(**base,queries=len(p),matched_paths=len(qidx),reference_paths=int(rk.sum()),predicted_paths=int(pk.sum()),recall=float(nm.sum()/rk.sum()) if rk.sum() else None,precision=float(nm.sum()/pk.sum()) if pk.sum() else None,miss_per_query=float((rk-nm).mean()),false_per_query=float((pk-nm).mean()),prediction_empty_fraction=float((pk==0).mean()),reference_empty_fraction=float((rk==0).mean()))
  for j,key in enumerate(['gain_db','delay_ns','aoa_direction_deg','aod_direction_deg']):
   for stat,v in [('rmse',np.sqrt(np.mean(err[:,j]**2)) if len(err) else None),('median',np.median(err[:,j]) if len(err) else None),('p90',np.quantile(err[:,j],.9) if len(err) else None)]:row[stat+'_'+key]=None if v is None else float(v)
  diff[:,1]*=1e9;diff[:,2:]*=180/np.pi
  for j,key in enumerate(['gain_db','delay_ns','aoa_az_deg','aoa_zen_deg','aod_az_deg','aod_zen_deg']):
   valid=np.ones(len(diff),bool)
   if j in [2,4]:valid=np.minimum(abs(np.sin(p[qidx,pi,j+1])),abs(np.sin(r[qidx,ri,j+1])))>m['azimuth_pole_sin_tolerance']
   row['six_rmse_'+key]=float(np.sqrt(np.mean(diff[valid,j]**2))) if valid.any() else None;row['six_support_'+key]=int(valid.sum())
  rows.append(row)
  for k in range(4):
   keep=rk==k;strata.append(dict(**base,reference_K=k,queries=int(keep.sum()),rms_gospa=float(np.sqrt(sq[keep].mean())),miss=float((rk-nm)[keep].mean()),false=float((pk-nm)[keep].mean())))
  if t['ratio']==1:
   all_errors.setdefault(t['variant'],[]).append(err)
   for sc in scales:
    cf=copy.deepcopy(cfg);cf['metric'].update(gain_scale_db=sc['gain_db'],delay_scale_s=sc['delay_ns']*1e-9,aoa_direction_scale_deg=sc['angle_deg'],aod_direction_scale_deg=sc['angle_deg'],gospa_c=sc['cutoff'])
    costs,_=matching(p,pk,r,rk,cf);sens.append(dict(**base,scale=sc['id'],rms_gospa=float(np.sqrt(costs.mean()))))
   if t['variant']=='Full' and t['pair']==5101:
    selected={}
    for label,ids,target in [('empty',np.flatnonzero(rk==0),0),('single',np.flatnonzero(rk==1),.5),('multiple',np.flatnonzero(rk>=2),.5),('median',np.arange(len(rk)),.5),('p90',np.arange(len(rk)),.9)]:
     value=np.quantile(sq[ids],target);qi=int(ids[0]) if label=='empty' else int(ids[np.argmin(abs(sq[ids]-value))]);selected[label]=dict(query_id=qi,prediction=p[qi,:pk[qi]].tolist(),reference=r[qi,:rk[qi]].tolist(),squared_gospa=float(sq[qi]),components=cc[qi].tolist())
    example_archive=dict(flow=5101,rule='Fixed first formal flow; closest squared-GOSPA to within-stratum median for single/multiple, lowest query ID for empty, pooled median/P90; ties lowest query ID. Includes errors, no success filtering.',selected=selected)
 assert scalar>=10000
 csvout(out/'physical_error_summary.csv',rows);csvout(out/'gospa_components_per_flow.csv',components);csvout(out/'reference_cardinality_results.csv',strata);csvout(out/'metric_sensitivity.csv',sens);csvout(out/'learning_curve_per_flow.csv',curves);save(out/'query_examples.json',example_archive)
 save(out/'validation.json',dict(source=str(source),protocol_hz=cfg['sounding']['delta_f_hz'],prototype_only=prototype,scalar_checks=scalar,scalar_checks_executed_this_stage=(scalar if prototype else 0),scalar_check_source=('this prototype stage' if prototype else 'results/final_validation.json'),scalar_max_error=squared_err,components_max_error=maxerr,empty_reference_rms=float(np.sqrt((rk*4.5).mean())),reference_empty_fraction=float((rk==0).mean()),created_at=stamp(),conditioning='Physical errors only on optimally matched paths against noiseless MUSIC-LS reference. Flow-level statistics; not independent cities.',no_matches='CSV blank = NA, support 0; precision NA for no predicted paths'))
 summaries=[]
 for v in NAMES:
  rr=[x for x in rows if x['variant']==v and x['ratio']==1];co=[x for x in components if x['variant']==v and x['ratio']==1]
  ss=dict(variant=v,display_name=NAMES[v],flows=len(rr),rms_gospa_mean=float(np.mean([x['rms_gospa'] for x in co])))
  for key in ['recall','precision','miss_per_query','false_per_query','rmse_gain_db','rmse_delay_ns','rmse_aoa_direction_deg','rmse_aod_direction_deg']:
   vv=[x[key] for x in rr if x[key] is not None];ss[key+'_flow_mean']=float(np.mean(vv)) if vv else None;ss[key+'_supported_flows']=len(vv)
  summaries.append(ss)
 csvout(out/'endpoint_summary.csv',summaries)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 labels=[NAMES[v] for v in NAMES];matrix=np.array([[np.mean([x[c] for x in components if x['variant']==v and x['ratio']==1]) for c in COMP] for v in NAMES])
 f,ax=plt.subplots(figsize=(9,4.3));left=np.zeros(5)
 for j,c in enumerate(COMP):ax.barh(np.arange(5),matrix[:,j],left=left,label=c);left+=matrix[:,j]
 ax.set_yticks(np.arange(5),labels);ax.invert_yaxis();ax.set_xlabel('Mean squared GOSPA contribution (query means, then flow means)');ax.legend(ncol=6,loc='upper center',bbox_to_anchor=(.5,-.17));f.tight_layout();f.savefig(fig/'gospa_components.png',dpi=180);plt.close(f)
 f,axs=plt.subplots(1,2,figsize=(10.3,4.2))
 for v in ['Coord','RawMap','Full']:
  rr=[.1,.25,.5,1];means=[];sd=[]
  for rat in rr:
   val=[x['rms_gospa'] for x in curves if x['variant']==v and x['ratio']==rat];means.append(np.mean(val));sd.append(np.std(val,ddof=1))
  axs[0].errorbar(np.array(rr)*100,means,yerr=sd,marker='o',label=NAMES[v],capsize=3)
 axs[0].set(xlabel='Training pool fraction (%)',ylabel='RMS GOSPA');axs[0].legend(fontsize=8)
 vals=[[x['rms_gospa'] for x in curves if x['variant']==v and x['ratio']==1] for v in NAMES]
 axs[1].errorbar(np.arange(5),np.mean(vals,1),yerr=np.std(vals,axis=1,ddof=1),fmt='o',capsize=3,color='#205493');axs[1].axhline(np.sqrt((rk*4.5).mean()),ls='--',color='#555',label='Always empty');axs[1].set_xticks(np.arange(5),['Coordinates','Direct map','Pooled geo.','No anchors','Proposed'],rotation=25,ha='right');axs[1].set_ylabel('RMS GOSPA');axs[1].legend(fontsize=8)
 f.suptitle(('Historical prototype: 500 kHz; ' if prototype else 'Revised protocol; ')+'error bars: SD across 10 flows',fontsize=11);f.tight_layout();f.savefig(fig/'learning_and_endpoints.png',dpi=180);plt.close(f)
 f,axs=plt.subplots(2,2,figsize=(9,6))
 for v,parts in all_errors.items():
  er=np.concatenate(parts)
  for j,ax in enumerate(axs.flat):
   if len(er):s=np.sort(er[:,j]);ax.plot(s,np.arange(1,len(s)+1)/len(s),label=f'{NAMES[v]} (n={len(er):,})',color=COLORS[v]);ax.set_xlabel(['Gain error (dB)','Delay error (ns)','AoA direction error (deg)','AoD direction error (deg)'][j]);ax.set_ylabel('Conditional matched-path CDF')
 axs[0,0].legend(fontsize=6.4);f.suptitle('Matched paths pooled over 10 flows; support varies by method',fontsize=10);f.tight_layout()
 for ext in ['png','pdf']:f.savefig(fig/f'physical_error_cdf.{ext}',dpi=180)
 plt.close(f)
 print(json.dumps(dict(stage='prototype' if prototype else 'final',groups=len(rows),validated_scalar_checks=scalar,scalar_replayed_here=(scalar if prototype else 0),components_max_error=maxerr)))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=EXP);ap.add_argument('--prototype',action='store_true');a=ap.parse_args();run(a.source,a.prototype)
