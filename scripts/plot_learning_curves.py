"""Five measured curves; single-column MATLAB style with embedded Times New Roman."""
from pathlib import Path
import sys
from paths import EXP, sha, save, stamp
import json
EXT=EXP/'extensions/full_learning_curves_20261030'
OUT=EXP/'figures'
OUT.mkdir(exist_ok=True)
def read(path):return json.loads(Path(path).read_text())
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

def main():
 assert read(EXT/'results/aggregation_validation.json')['passed']
 font=font_manager.findfont('Times New Roman',fallback_to_default=False)
 plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.linewidth':.65,'axes.edgecolor':'black','axes.spines.top':True,'axes.spines.right':True,'xtick.direction':'in','ytick.direction':'in','xtick.major.width':.65,'ytick.major.width':.65,'grid.color':'#bdbdbd','grid.alpha':.5,'grid.linestyle':':','grid.linewidth':.5,'legend.framealpha':1,'legend.fancybox':False,'legend.edgecolor':'black','pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white','axes.unicode_minus':False})
 palette=['#0072BD','#D95319','#EDB120','#7E2F8E','#77AC30']
 colors=dict(Full=palette[0],RawMap=palette[1],Coord=palette[2],GeoPool=palette[3],GeoDirect=palette[4])
 names=dict(Coord='Coord',RawMap='RawMap',GeoPool='GeoPool',GeoDirect='GeoDirect',Full='Proposed')
 with (EXT/'results/summary_20_cells.csv').open() as f:summary=list(csv.DictReader(f))
 assert len(summary)==20
 fig,ax=plt.subplots(figsize=(3.5,2.65));fig.subplots_adjust(left=.145,right=.975,top=.97,bottom=.19)
 plotted=[]
 for variant,marker,line in [('Coord','s','--'),('RawMap','^','-.'),('GeoPool','D',':'),('GeoDirect','v','--'),('Full','o','-')]:
  rows=sorted([r for r in summary if r['variant']==variant],key=lambda r:float(r['ratio']))
  assert [float(r['ratio']) for r in rows]==[.1,.25,.5,1.]
  x=np.array([float(r['n_train_mean']) for r in rows]);y=np.array([float(r['gospa_mean']) for r in rows]);sd=np.array([float(r['gospa_sd']) for r in rows])
  ax.errorbar(x,y,yerr=sd,color=colors[variant],marker=marker,linestyle=line,markersize=3.6,markerfacecolor='white',markeredgewidth=.85,linewidth=1.1,elinewidth=.8,capsize=2,capthick=.8,label=names[variant])
  plotted.extend(dict(variant=variant,display_name=names[variant],ratio=float(r['ratio']),x=float(xx),mean=float(yy),sd=float(ss)) for r,xx,yy,ss in zip(rows,x,y,sd))
 low=min(r['mean']-r['sd'] for r in plotted);high=max(r['mean']+r['sd'] for r in plotted)
 ax.set(xlim=(650,10000),ylim=(np.floor((low-.03)*10)/10,np.ceil((high+.03)*10)/10),xlabel='Training observations',ylabel='RMS GOSPA')
 ax.set_xticks([1000,3000,5000,7000,9000]);ax.grid(True);ax.set_axisbelow(True)
 handles,labels=ax.get_legend_handles_labels();ix=[0,3,1,4,2]
 legend=ax.legend([handles[i] for i in ix],[labels[i] for i in ix],loc='center',bbox_to_anchor=(.51,.56),ncol=3,handlelength=1.5,handletextpad=.35,columnspacing=.65,borderpad=.45)
 legend.get_frame().set_linewidth(.6)
 stem='five_method_learning_curves'
 for ext in ['pdf','png']:fig.savefig(OUT/f'{stem}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.015)
 plt.close(fig)
 with (OUT/'plotted_data.csv').open('w',newline='') as out:
  w=csv.DictWriter(out,fieldnames=list(plotted[0]));w.writeheader();w.writerows(plotted)
 save(OUT/'source_manifest.json',dict(created_at=stamp(),font='Times New Roman',font_file=Path(font).name,font_sha256=sha(font),display_names=names,style='MATLAB ColorOrder, boxed axes, inward ticks, distinct markers/linestyles',error_bars='Sample SD (ddof=1) across 10 paired flows',x_axis='Actual training-record counts averaged across flows for each frozen nested subset',five_real_curves=True,no_endpoint_subplot=True,single_column_width_inches=3.5,sources={(EXT/'results/summary_20_cells.csv').relative_to(EXP).as_posix():sha(EXT/'results/summary_20_cells.csv'),(EXT/'results/per_flow_200.csv').relative_to(EXP).as_posix():sha(EXT/'results/per_flow_200.csv')},plot_script_sha256=sha(__file__),outputs={name:sha(OUT/name) for name in [stem+'.pdf',stem+'.png','plotted_data.csv']}))
 print('Five-method figures generated with',font)
if __name__=='__main__':main()
