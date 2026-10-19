from common import *
from plot_style import setup,savefig,COLORS

def run():
    plt=setup();paired=list(csv.DictReader((EXP/'results/paired_scale_summary.csv').open()));rounds=list(csv.DictReader((EXP/'results/scale_grid_summary.csv').open()));groups=list(csv.DictReader((EXP/'results/group_summary.csv').open()));grid=scales();ids=[r['id'] for r in grid];y=np.arange(15)
    fig,axes=plt.subplots(1,2,figsize=(10.5,6.3),sharey=True)
    for ax,name,title,color in zip(axes,['geometry','cooperation'],['Geometric learner: I-G / I-D','Cooperative coverage: C-G / I-G'],COLORS[:2]):
        rr=[next(r for r in paired if r['scale']==s and r['comparison']==name) for s in ids];mean=np.array([float(r['mean']) for r in rr]);low=np.array([float(r['lo']) for r in rr]);high=np.array([float(r['hi']) for r in rr]);ax.errorbar(mean,y,xerr=np.array([mean-low,high-mean]),fmt='o',capsize=3,color=color,ms=4);ax.axvline(0,color='black',lw=.7);ax.set(title=title,xlabel='Paired terminal GOSPA reduction (%)',yticks=y,yticklabels=ids);ax.grid(axis='x',alpha=.18)
    axes[0].invert_yaxis();fig.suptitle('Fixed predictions: 10 paired seeds, mean and unadjusted 95% t interval',fontweight='bold',fontsize=12);fig.tight_layout();savefig(fig,'scale_paired_effects')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8));times=np.arange(30,241,30)
    for ax,scale,title in zip(axes,['S0','S_phys'],['S0: 5 dB / 25 ns / 15°','Physical reference: 5 dB / 15.625 ns / 31.54449°']):
        for b,color in zip(BRANCHES[:3],COLORS[:3]):
            rr=sorted([r for r in rounds if r['scale']==scale and r['branch']==b],key=lambda r:float(r['round']));mean=np.array([float(r['global_gospa_mean']) for r in rr]);sd=np.array([float(r['global_gospa_sd']) for r in rr]);margin=2.2621571628540993*sd/np.sqrt(10);ax.plot(times,mean,'o-',label=b,color=color,ms=4);ax.fill_between(times,mean-margin,mean+margin,color=color,alpha=.13)
        ax.set(xlabel='Net sensing time (min)',ylabel='Global GOSPA under panel scale',title=title,xticks=times);ax.legend();ax.grid(alpha=.18)
    fig.suptitle('Compare methods within each scale; vertical values use different metrics',fontstyle='italic',fontsize=11);fig.tight_layout();savefig(fig,'S0_vs_physical_learning_curves')
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8));bottom=np.zeros(8);labels=[f'{b}\n{s}' for s in ['S0','S_phys'] for b in BRANCHES];lookup={(r['scale'],r['branch'],r['subset']):r for r in groups if float(r['round'])==7};costs=[]
    for scale in ['S0','S_phys']:
        for b in BRANCHES:costs.append([float(lookup[scale,b,'all'][key+'_mean']) for key in ['matched_parameter_cost','miss_cost','false_cost']])
    costs=np.array(costs);fractions=costs/costs.sum(-1,keepdims=True)
    for j,label,color in zip(range(3),['Matched parameters','Missed components','False components'],COLORS[:3]):axes[0].bar(np.arange(8),fractions[:,j],bottom=bottom,color=color,label=label);bottom+=fractions[:,j]
    axes[0].set(xticks=np.arange(8),xticklabels=labels,ylabel='Fraction of within-scale squared cost',ylim=(0,1.01),title='Terminal cost composition');axes[0].tick_params(axis='x',labelsize=8);axes[0].legend(fontsize=9)
    categories=['all','K2plus','second_related'];x=np.arange(3)
    for j,(b,color) in enumerate(zip(BRANCHES,COLORS[:4])):
        values=[100*float(lookup['S_phys',b,g]['assignment_changed_fraction_mean']) for g in categories];axes[1].bar(x+(j-1.5)*.18,values,.18,label=b,color=color)
    axes[1].set(xticks=x,xticklabels=['All queries','K >= 2','Second-related'],ylabel='Queries with changed assignment (%)',title='S_phys matching versus S0');axes[1].legend(fontsize=9)
    for ax in axes:ax.grid(axis='y',alpha=.18)
    fig.tight_layout();savefig(fig,'matching_and_error_composition');print('THREE_SENSITIVITY_FIGURES_CREATED',flush=True)

if __name__=='__main__':run()
