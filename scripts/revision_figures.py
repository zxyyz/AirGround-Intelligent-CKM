"""Publication figures and supplemental illustrations from final CSVs only."""
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from paths import EXP
from revision_metrics import NAMES
SHORT={'Coord':'Coordinates','RawMap':'Direct map','GeoPool':'Pooled geometry','GeoDirect':'No anchors','Full':'Proposed'}
COLORS={'Coord':'#b34a3d','RawMap':'#bf851c','GeoPool':'#7d65a5','GeoDirect':'#418c74','Full':'#176b9c'}
def read(name):
 with (EXP/'results'/name).open() as f:return list(csv.DictReader(f))
def savefig(fig,name):
 for ext in ['pdf','png']:fig.savefig(EXP/f'figures/final/{name}.{ext}',dpi=250,bbox_inches='tight',pad_inches=.04)
 plt.close(fig)
def main():
 (EXP/'figures/final').mkdir(exist_ok=True);rows=read('summary.csv');stats=json.loads((EXP/'results/statistical_summary.json').read_text())
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(2,1,figsize=(3.45,3.8),gridspec_kw={'height_ratios':[1.1,1]})
 for v,marker in [('Coord','s'),('RawMap','^'),('Full','o')]:
  rr=sorted([r for r in rows if r['variant']==v],key=lambda x:float(x['ratio']));x=[float(r['n_train_mean']) for r in rr];y=[float(r['gospa_mean']) for r in rr];error=[float(r['gospa_sd']) for r in rr]
  axs[0].errorbar(x,y,yerr=error,lw=1.1,marker=marker,ms=3,color=COLORS[v],capsize=2,label=SHORT[v])
 axs[0].set(xlabel='Training observations per flow',ylabel='RMS GOSPA');axs[0].legend(fontsize=6.5,ncol=3,loc='center',bbox_to_anchor=(.5,.6),handlelength=1.4,columnspacing=.8);axs[0].text(0,1.04,'(a) Three measured learning curves',transform=axs[0].transAxes,fontweight='bold',fontsize=8)
 vlist=list(SHORT);rr=[next(r for r in rows if r['variant']==v and float(r['ratio'])==1) for v in vlist]
 for i,(v,r) in enumerate(zip(vlist,rr)):axs[1].errorbar(i,float(r['gospa_mean']),yerr=float(r['gospa_sd']),fmt='o',ms=4,capsize=3,color=COLORS[v])
 axs[1].axhline(stats['all_empty_gospa'],ls='--',lw=.8,color='#555',label='Always empty');axs[1].set_xticks(range(5),['Coordinates','Direct\nmap','Pooled\ngeometry','No\nanchors','Proposed'],fontsize=7);axs[1].set_ylabel('RMS GOSPA');axs[1].legend(fontsize=7,loc='lower left');axs[1].text(0,1.05,'(b) All five methods at 100% data',transform=axs[1].transAxes,fontweight='bold',fontsize=8)
 for ax in axs:ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
 fig.text(.5,.005,'Error bars: SD across 10 paired flows',ha='center',fontsize=7)
 fig.tight_layout(h_pad=1.1,rect=(0,.035,1,1));savefig(fig,'learning_and_endpoints')
 comp=read('final/gospa_components_per_flow.csv');labels=['Gain','Delay','AoA','AoD','Missed','False'];cols=['gain','delay','aoa','aod','miss','false'];mat=np.array([[np.mean([float(r[c]) for r in comp if r['variant']==v and float(r['ratio'])==1]) for c in cols] for v in vlist])
 fig,ax=plt.subplots(figsize=(6.7,2.8));left=np.zeros(5)
 for j,lab in enumerate(labels):ax.barh(range(5),mat[:,j],left=left,label=lab);left+=mat[:,j]
 ax.set_yticks(range(5),[SHORT[v] for v in vlist]);ax.invert_yaxis();ax.set_xlabel('Mean squared GOSPA; components add before taking the square root');ax.legend(ncol=6,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.22));fig.tight_layout();savefig(fig,'squared_gospa_decomposition')
 examples=json.loads((EXP/'results/final/query_examples.json').read_text());q=np.load(EXP/'cache/queries_confirm.npz')['queries']
 fig,axs=plt.subplots(1,5,figsize=(11,2.8),sharey=True)
 for ax,(kind,case) in zip(axs,examples['selected'].items()):
  for label,key,col,m in [('Reference','reference','#222','x'),('Proposed','prediction','#176b9c','o')]:
   z=np.asarray(case[key]).reshape(-1,6)
   if len(z):ax.scatter(z[:,1]*1e9,z[:,0],c=col,marker=m,s=28,label=label)
  if not len(case['reference']) and not len(case['prediction']):ax.text(.5,.5,'Both empty',transform=ax.transAxes,ha='center');ax.set_xticks([])
  else:ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
  if len(case['reference']) and not len(case['prediction']):ax.text(.5,.15,'No predicted paths',transform=ax.transAxes,ha='center',fontsize=7)
  ax.set_title(f'{kind.title()} | query {case["query_id"]}\nGOSPA = {np.sqrt(case["squared_gospa"]):.3f}',fontsize=8);ax.set_xlabel('Delay (ns)');ax.grid(alpha=.2);case['query_coordinates_m']=q[case['query_id']].tolist()
 axs[0].set_ylabel('Path power gain (dB)');handles,labels=next((ax.get_legend_handles_labels() for ax in axs if ax.get_legend_handles_labels()[0]),([],[]));fig.legend(handles,labels,loc='upper center',ncol=2,fontsize=8);fig.tight_layout(rect=(0,0,1,.9));savefig(fig,'fixed_query_examples');(EXP/'results/final/query_examples.json').write_text(json.dumps(examples,indent=2))
 sens=read('scale_sensitivity_contrasts.csv');scales=[s['id'] for s in json.loads((EXP/'configs/supplementary_protocol.json').read_text())['scales']]
 fig,ax=plt.subplots(figsize=(8,3.4))
 for treatment,control,mark in [('RawMap','Coord','^'),('GeoPool','RawMap','D'),('Full','RawMap','o'),('Full','GeoDirect','s')]:
  rr=[next(r for r in sens if r['scale']==s and r['treatment']==treatment and r['control']==control) for s in scales];ax.plot(range(15),[float(r['mean_relative_improvement_percent']) for r in rr],marker=mark,ms=3,label=f'{SHORT[treatment]} vs {SHORT[control]}')
 ax.axhline(0,color='#555',lw=.7);ax.set_xticks(range(15),scales,rotation=40,ha='right');ax.set_ylabel('Mean paired relative improvement (%)');ax.legend(fontsize=8,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.28));ax.grid(axis='y',alpha=.2);fig.tight_layout();savefig(fig,'all_scale_sensitivity')
if __name__=='__main__':main()
