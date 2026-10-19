from common import *
from scipy import stats
from collections import defaultdict

COMPARISONS=[('geometry','I-G','I-D'),('cooperation','C-G','I-G'),('disagreement','P-G','C-G')]

def interval(a):
    a=np.asarray(a,float);mean=float(a.mean());sd=float(a.std(ddof=1));margin=stats.t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a));p=float(stats.ttest_1samp(a,0).pvalue) if sd else (0. if mean else 1.)
    return dict(mean=mean,sd=sd,lo=mean-margin,hi=mean+margin,p=p,n=len(a),wins=int((a>0).sum()),losses=int((a<0).sum()))

def holm(pvalues):
    order=np.argsort(pvalues);out=np.zeros(len(pvalues));v=0.
    for rank,i in enumerate(order):v=max(v,min(1.,(len(pvalues)-rank)*pvalues[i]));out[i]=v
    return out

def read_rows():
    rows=[]
    for pair in range(5101,5111):
        for row in csv.DictReader((EXP/f'results/scale_rows_{pair}.csv').open()):
            rows.append({k:(v if k in ['scale','branch','subset'] else float(v) if v else np.nan) for k,v in row.items()})
    return rows

def run():
    start=time.perf_counter();rows=read_rows();allrows=[r for r in rows if r['subset']=='all'];assert len(allrows)==4800
    lookup={(r['scale'],int(r['pair']),r['branch'],int(r['round'])):r for r in allrows};grid=scales();scaleids=[r['id'] for r in grid];paired=[];summaries=[];endpoints=[r for r in allrows if r['round']==7];groups=defaultdict(list);roundsummary=[];groupsummary=[]
    for r in rows:groups[r['scale'],r['branch'],int(r['round']),r['subset']].append(r)
    metricnames=['global_gospa','mean_squared_gospa','matched_parameter_cost','miss_cost','false_cost','matched_recall','second_matched_recall','mean_miss','mean_false','n_matched','n_missed','n_false','count_accuracy','assignment_changed_fraction','assignment_changed_active_fraction']
    for (scale,branch,r,subset),rr in groups.items():
        summary=dict(scale=scale,branch=branch,round=r,minutes=(r+1)*30,subset=subset,n_seeds=10,n_queries=int(rr[0]['n_queries']))
        for key in metricnames:
            a=np.array([d.get(key,np.nan) for d in rr]);summary[key+'_mean']=float(a.mean());summary[key+'_sd']=float(a.std(ddof=1))
        groupsummary.append(summary)
        if subset=='all':roundsummary.append(summary)
    for scale in scaleids:
        this=[]
        for name,method,control in COMPARISONS:
            relative=[];absolute=[];aucrel=[]
            for pair in range(5101,5111):
                a=lookup[scale,pair,method,7]['global_gospa'];b=lookup[scale,pair,control,7]['global_gospa'];improve=100*(1-a/b);diff=b-a
                aa=float(np.trapezoid([lookup[scale,pair,method,r]['mean_squared_gospa'] for r in range(8)],np.arange(30,241,30))/210);ab=float(np.trapezoid([lookup[scale,pair,control,r]['mean_squared_gospa'] for r in range(8)],np.arange(30,241,30))/210)
                relative.append(improve);absolute.append(diff);aucrel.append(100*(1-aa/ab));paired.append(dict(scale=scale,comparison=name,pair=pair,method=method,control=control,method_terminal_gospa=a,control_terminal_gospa=b,improvement_percent=improve,absolute_improvement=diff,method_mean_squared_auc=aa,control_mean_squared_auc=ab,auc_improvement_percent=100*(1-aa/ab)))
            value=dict(scale=scale,comparison=name,method=method,control=control,primary=name!='disagreement',**interval(relative),absolute_mean=float(np.mean(absolute)),absolute_lo=interval(absolute)['lo'],absolute_hi=interval(absolute)['hi'],auc_improvement_mean=float(np.mean(aucrel)),auc_improvement_lo=interval(aucrel)['lo'],auc_improvement_hi=interval(aucrel)['hi']);this.append(value)
        adjusted=holm([v['p'] for v in this[:2]])
        for row,p in zip(this[:2],adjusted):row['holm_p_within_scale']=float(p);row['supported_positive_within_scale']=bool(p<.05 and row['mean']>0)
        summaries.extend(this)
    baseline={r['comparison']:r for r in summaries if r['scale']=='S0'}
    for r in summaries:r['change_from_S0_pp']=r['mean']-baseline[r['comparison']]['mean']
    csvwrite(EXP/'results/scale_grid_summary.csv',roundsummary);csvwrite(EXP/'results/group_summary.csv',groupsummary);csvwrite(EXP/'results/paired_scale_effects.csv',paired);csvwrite(EXP/'results/paired_scale_summary.csv',summaries);csvwrite(EXP/'results/endpoint_by_scale.csv',endpoints);csvwrite(EXP/'results/scale_error_breakdown.csv',allrows);csvwrite(EXP/'results/group_sensitivity.csv',rows)
    # Count invariance must hold across all scales, including group counts.
    invariant={}
    for r in rows:
        key=(r['pair'],r['branch'],r['round'],r['subset']);value=tuple(r[k] for k in ['count_accuracy','predicted_components','reference_components','n_queries'])
        assert key not in invariant or value==invariant[key];invariant[key]=value
    robustness={}
    for name,_,_ in COMPARISONS:
        rr=[r for r in summaries if r['comparison']==name]
        robustness[name]=dict(mean_range_percent=[min(r['mean'] for r in rr),max(r['mean'] for r in rr)],all_scale_mean_positive=all(r['mean']>0 for r in rr),all_scale_CI_lower_positive=all(r['lo']>0 for r in rr),all_seed_positive_every_scale=all(r['wins']==10 for r in rr),CI_crossing_zero_scales=[r['scale'] for r in rr if r['lo']<=0<=r['hi']],mean_ranking_reversal_scales=[r['scale'] for r in rr if r['mean']<0],effect_change_gt5pp_scales=[r['scale'] for r in rr if abs(r['change_from_S0_pp'])>5],maximum_abs_change_from_S0_pp=max(abs(r['change_from_S0_pp']) for r in rr))
    assignment={}
    for branch in BRANCHES:
        rr=[r for r in groupsummary if r['branch']==branch and r['round']==7 and r['subset'] in ['all','K2plus','second_related']]
        assignment[branch]={group:max((r for r in rr if r['subset']==group),key=lambda a:a['assignment_changed_fraction_mean']) for group in ['all','K2plus','second_related']}
    result=dict(epoch=time.time(),passed=True,full_scale_updates=4800,seed_unit=10,counts_invariant=True,robustness=robustness,endpoint_assignment_maxima=assignment,within_scale_multiple_testing='two main endpoints Holm; across-scale statements descriptive correlated sensitivity',wall_s=time.perf_counter()-start)
    save(EXP/'results/sensitivity_summary.json',result);stage('C_complete_D1_pending',C='passed');print(json.dumps(robustness,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    try:run()
    except Exception as e:failure('analysis',e);raise
