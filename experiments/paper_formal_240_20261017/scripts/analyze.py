"""Prespecified paired statistics, paper plots, and sole human ledger update."""
from common import *
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BRANCHES=['I-D','I-G','C-G','P-G']
COMPARISONS=[('geometry','I-G','I-D'),('cooperation','C-G','I-G'),('disagreement','P-G','C-G'),('complete_vs_independent','P-G','I-G')]

def interval(values):
    a=np.asarray(values,float);mean=float(a.mean());sd=float(a.std(ddof=1));margin=float(stats.t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a)));p=float(stats.ttest_1samp(a,0).pvalue) if sd>0 else (0. if mean else 1.)
    return dict(n=len(a),mean=mean,sd=sd,t95=[mean-margin,mean+margin],two_sided_p=p,values=a.tolist())

def holm(values):
    order=np.argsort(values);out=np.zeros(len(values));current=0.
    for rank,i in enumerate(order):current=max(current,min(1.,(len(values)-rank)*values[i]));out[i]=current
    return out.tolist()

def load_all():
    rounds=[];subsets=[];costs=[];lookup={}
    for pair in range(5101,5111):
        root=EXP/f'runs/formal/{pair}';assert (root/'evaluation_complete.json').exists()
        for branch in BRANCHES:
            lookup[pair,branch]=[]
            for r in range(8):
                d=json.loads((root/branch/f'metrics_{r:02d}.json').read_text());allm=next(a for a in d['metrics'] if a['subset']=='all');base={k:d[k] for k in ['pair','branch','variant','round','minutes','n_attempts','threshold']};row=dict(**base,**allm);rounds.append(row);lookup[pair,branch].append(row)
                subsets.extend(dict(**base,**m) for m in d['metrics'])
                meta=json.loads((root/branch/f'checkpoints/update_{r:02d}.json').read_text()) if r else json.loads((root/f'initial_{"D" if branch=="I-D" else "M3"}/checkpoints/update_00.json').read_text())
                costs.append(dict(**base,**meta['timing'],inference_s=d['inference_s'],scoring_s=d['scoring_s']))
    return rounds,subsets,costs,lookup

def run():
    rounds,subsets,costs,lookup=load_all();csvwrite(EXP/'results/round_metrics.csv',rounds);csvwrite(EXP/'results/subset_metrics.csv',subsets);csvwrite(EXP/'results/compute_cost.csv',costs)
    endpoints=[lookup[p,b][-1] for p in range(5101,5111) for b in BRANCHES];csvwrite(EXP/'results/endpoint.csv',endpoints);summaries={};paired=[]
    for name,new,control in COMPARISONS:
        relative=[];absolute=[];aucrel=[];second=[]
        for pair in range(5101,5111):
            a=lookup[pair,new];b=lookup[pair,control];e=a[-1]['global_gospa'];base=b[-1]['global_gospa'];r=100*(1-e/base);diff=base-e
            aa=float(np.trapezoid([q['global_gospa']**2 for q in a],np.arange(30,241,30))/210);ab=float(np.trapezoid([q['global_gospa']**2 for q in b],np.arange(30,241,30))/210)
            relauc=100*(1-aa/ab);relative.append(r);absolute.append(diff);aucrel.append(relauc);second.append(100*(a[-1]['second_matched_recall']-b[-1]['second_matched_recall']))
            paired.append(dict(comparison=name,pair=pair,method=new,control=control,method_global_gospa=e,control_global_gospa=base,relative_improvement_percent=r,absolute_improvement=diff,method_mean_squared_auc=aa,control_mean_squared_auc=ab,auc_improvement_percent=relauc))
        summaries[name]=dict(method=new,control=control,endpoint_relative=interval(relative),endpoint_absolute=interval(absolute),mean_squared_auc_relative=interval(aucrel),second_recall_gain_pp=interval(second),primary=name in ['geometry','cooperation'])
    adjusted=holm([summaries[k]['endpoint_relative']['two_sided_p'] for k in ['geometry','cooperation']])
    for k,p in zip(['geometry','cooperation'],adjusted):summaries[k]['holm_p']=p;summaries[k]['supported_primary_positive']=bool(p<.05 and summaries[k]['endpoint_relative']['mean']>0)
    csvwrite(EXP/'results/paired_comparisons.csv',paired);save(EXP/'results/paired_summary.json',dict(seeds=list(range(5101,5111)),single_fixed_city=True,primary_family_alpha=.05,comparisons=summaries))
    figures(rounds,subsets,lookup,paired)
    write_ledger(summaries,endpoints)
    print(json.dumps({k:v['endpoint_relative'] for k,v in summaries.items()},ensure_ascii=False,indent=2),flush=True)

def figures(rounds,subsets,lookup,paired):
    dest=EXP/'figures';dest.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
    colors={'I-D':'#666666','I-G':'#2c7bb6','C-G':'#1a9850','P-G':'#d95f02'};times=np.arange(30,241,30)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for b in BRANCHES:
        a=np.array([[lookup[p,b][r]['global_gospa'] for r in range(8)] for p in range(5101,5111)]);mean=a.mean(0);margin=stats.t.ppf(.975,9)*a.std(0,ddof=1)/np.sqrt(10)
        axes[0].plot(times,mean,'o-',label=b,color=colors[b]);axes[0].fill_between(times,mean-margin,mean+margin,color=colors[b],alpha=.12)
        if b!='I-D':axes[1].plot(times,mean,'o-',label=b,color=colors[b]);axes[1].fill_between(times,mean-margin,mean+margin,color=colors[b],alpha=.12)
    for a in axes:a.set_xlabel('Net sensing time (min)');a.set_ylabel('Global GOSPA');a.set_xticks(times);a.grid(alpha=.2);a.legend()
    axes[0].set_title('All mapping and acquisition configurations');axes[1].set_title('M3 acquisition comparison (detail)');fig.tight_layout();savefig(fig,dest/'01_learning_curves')
    fig,ax=plt.subplots(figsize=(7,3.8));names=['geometry','cooperation','disagreement'];labels=['I-G / I-D','C-G / I-G','P-G / C-G']
    for j,name in enumerate(names):
        values=np.array([r['relative_improvement_percent'] for r in paired if r['comparison']==name]);summary=interval(values);ax.scatter(np.linspace(j-.11,j+.11,10),values,color=['#2c7bb6','#1a9850','#d95f02'][j],alpha=.7,s=24);ax.errorbar(j,summary['mean'],yerr=summary['mean']-summary['t95'][0],fmt='ks',capsize=5)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(3),labels);ax.set_ylabel('Paired terminal GOSPA reduction (%)');ax.set_title('Ten prespecified seeds; mean and unadjusted 95% t interval');ax.grid(axis='y',alpha=.2);fig.tight_layout();savefig(fig,dest/'02_paired_effects')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8));groups=['nonempty','K2plus','second_related'];x=np.arange(len(groups));width=.19
    for j,b in enumerate(BRANCHES):
        values=[np.mean([r['global_gospa'] for r in subsets if r['round']==7 and r['branch']==b and r['subset']==group]) for group in groups];axes[0].bar(x+(j-1.5)*width,values,width,label=b,color=colors[b])
    axes[0].set_xticks(x,['Nonempty','K >= 2','Second-related']);axes[0].set_ylabel('Terminal GOSPA');axes[0].legend(fontsize=8)
    bottom=np.zeros(4)
    for key,label,color in [('matched_parameter_cost','Matched parameters','#7570b3'),('miss_cost','Missed components','#e7298a'),('false_cost','False components','#66a61e')]:
        values=np.array([np.mean([r[key] for r in rounds if r['round']==7 and r['branch']==b]) for b in BRANCHES]);axes[1].bar(np.arange(4),values,bottom=bottom,label=label,color=color);bottom+=values
    axes[1].set_xticks(np.arange(4),BRANCHES);axes[1].set_ylabel('Mean squared GOSPA cost');axes[1].legend(fontsize=8)
    for a in axes:a.grid(axis='y',alpha=.2)
    fig.tight_layout();savefig(fig,dest/'03_subsets_and_errors')

def savefig(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=180);fig.savefig(path.with_suffix('.pdf'));plt.close(fig)

def write_ledger(summary,endpoints):
    ledger=ROOT/'experiments/geometry_conditioned_path_detection_20261017/FINAL_REPORT_CN.md';s=ledger.read_text();title='## 论文正式全量仿真结果（M3/W1a，10组配对seed）'
    if title in s:s=s[:s.index(title)]+s[s.index('## 下一轮：论文正式全量仿真计划',s.index(title)):]
    text=title+'\n\n正式seed5101–5110、每组四配置、8个完整30 min窗口均已完成。每分支14400次探测预算、7100步/成员；4096个自然分布查询只供独立离线评价。以下结果与上方120 min探索性结果分开。\n\n'
    text+='| 配置 | 240 min全局GOSPA（均值±SD） | 二次关联分量召回（均值） |\n| --- | --- | --- |\n'
    for b in BRANCHES:
        rows=[r for r in endpoints if r['branch']==b];a=np.array([r['global_gospa'] for r in rows]);rec=np.mean([r['second_matched_recall'] for r in rows]);text+=f'| {b} | {a.mean():.4f} ± {a.std(ddof=1):.4f} | {100*rec:.2f}% |\n'
    text+='\n| 比较 | 逐seed终点改善均值 | 未校正95%配对t区间 | Holm校正p（仅两个主比较） |\n| --- | --- | --- | --- |\n'
    for name,_,_ in COMPARISONS:
        r=summary[name];v=r['endpoint_relative'];p=f"{r['holm_p']:.5g}" if 'holm_p' in r else '次要比较';text+=f"| {r['method']} / {r['control']} | {v['mean']:.2f}% | [{v['t95'][0]:.2f}, {v['t95'][1]:.2f}]% | {p} |\n"
    text+='\n'+('几何建图主比较支持正向收益。' if summary['geometry']['supported_primary_positive'] else '几何建图主比较未达到预定的正向统计支持条件。')+(' 同M3下，协同纯覆盖主比较支持优于指定独立覆盖。' if summary['cooperation']['supported_primary_positive'] else ' 同M3下，协同纯覆盖主比较未达到预定的正向统计支持条件，论文应据此收缩协同收益主张。')+' 分歧比较作为次要描述性结果，不据其表现追加实验。\n\n'
    text+='结果仅适用于这个固定城市及物理/前端规则；4096查询不是4096次独立试验。I-D含G0公开几何输入且使用统一空间重放，I-G/I-D反映整个几何候选级方案，单模块解释沿用既有同数据消融。\n\n'
    for name,alt in [('01_learning_curves','正式误差曲线'),('02_paired_effects','正式配对终点收益'),('03_subsets_and_errors','非空、多径和误差分解')]:text+=f'![{alt}]({EXP}/figures/{name}.png)\n\n'
    text+=f'机器证据：[冻结配置]({EXP}/configs/freeze.json)、[配对统计]({EXP}/results/paired_summary.json)、[完整性审计]({EXP}/results/integrity.json)。正式论文与Overleaf本轮未改。\n\n'
    marker='## 下一轮：论文正式全量仿真计划';s=s.replace(marker,text+marker,1);ledger.write_text(s)

if __name__=='__main__':
    try:run()
    except Exception as e:failure('analyze',e);raise
