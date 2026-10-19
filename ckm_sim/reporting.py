"""All curves, tables and manuscript numbers derive from persisted run CSVs."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import ROOT,write_json,load_config

def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def save_csv(path,data):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)

def report(runroot):
    runroot=Path(runroot).resolve();is_main=runroot.name=='main'
    output=ROOT/'reports'/runroot.name;output.mkdir(parents=True,exist_ok=True)
    manifests=list(runroot.glob('*/*/manifest.json'))
    if not manifests:raise ValueError('No branch manifests')
    data=[];checks=[];seeds=[]
    for path in sorted(manifests):
        m=json.loads(path.read_text());cfg=m['config'];seed=m['pair_seed'];policy=m['policy']
        branch=rows(path.parent/'round_metrics.csv');data.extend(branch);seeds.append(seed)
        expected=cfg['experiment']['rounds_after_initial']+1
        ok=m['status']=='complete' and len(branch)==expected
        for r,row in enumerate(branch):
            ok &= int(row['round'])==r and float(row['sensing_time_s'])==cfg['experiment']['initial_sensing_time_s']+r*cfg['experiment']['round_time_s']
            ok &= int(row['n_attempts'])==round(float(row['sensing_time_s'])*cfg['experiment']['probe_frequency_hz'])
            ok &= int(row['n_train'])==int(row['n_valid'])
        checks.append({'pair_seed':seed,'policy':policy,'passed':bool(ok),'rounds':len(branch),
            'initial_records_hash':m['common_initial_records_hash'],'initial_weights_hash':m['common_initial_weights_hash']})
    seeds=sorted(set(seeds))
    for seed in seeds:
        pair=[x for x in checks if x['pair_seed']==seed]
        if len(pair)!=2 or len({x['initial_records_hash'] for x in pair})!=1 or len({x['initial_weights_hash'] for x in pair})!=1:
            raise ValueError('Paired common initialization differs')
    flow=all(x['passed'] for x in checks)
    if is_main and (len(checks)!=6 or seeds!=[2101,2102,2103]):flow=False
    save_csv(output/'per_seed_round_metrics.csv',data)
    final_round=max(int(row['round']) for row in data);final=[r for r in data if int(r['round'])==final_round]
    save_csv(output/'per_seed_final_metrics.csv',final)
    summaries=[]
    numeric=[k for k in final[0] if k not in ['policy','pair_seed','round']]
    for policy in ['independent','proposed']:
        selected=[r for r in final if r['policy']==policy];summary={'policy':policy,'n_pairs':len(selected)}
        for key in numeric:
            v=np.array([float(r[key]) for r in selected]);finite=v[np.isfinite(v)]
            summary[key+'_mean']=float(finite.mean()) if len(finite) else np.nan
            summary[key+'_std']=float(finite.std(ddof=1)) if len(finite)>1 else (0. if len(finite) else np.nan)
        summaries.append(summary)
    save_csv(output/'final_metrics.csv',summaries)
    paired=[]
    for seed in seeds:
        a=float(next(r['global_gospa'] for r in final if int(r['pair_seed'])==seed and r['policy']=='independent'))
        b=float(next(r['global_gospa'] for r in final if int(r['pair_seed'])==seed and r['policy']=='proposed'))
        paired.append({'pair_seed':seed,'independent_gospa':a,'proposed_gospa':b,'absolute_improvement':a-b,
                       'relative_improvement':(a-b)/a if a>0 else np.nan})
    save_csv(output/'paired_final_differences.csv',paired)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(6.7,3.65),constrained_layout=True)
    for policy,color,label in [('independent','#3972a6','Independent'),('proposed','#c65136','Proposed')]:
        curves=[]
        for seed in seeds:
            curve=sorted([r for r in data if int(r['pair_seed'])==seed and r['policy']==policy],key=lambda r:int(r['round']))
            t=np.array([float(r['sensing_time_s'])/60 for r in curve]);y=np.array([float(r['global_gospa']) for r in curve]);curves.append(y)
            ax.plot(t,y,color=color,alpha=.22,lw=.8)
        curves=np.array(curves);mean=curves.mean(0);std=curves.std(0,ddof=1) if len(curves)>1 else np.zeros(len(t))
        ax.plot(t,mean,'o-',color=color,label=label,lw=1.9,markersize=3.5)
        ax.fill_between(t,mean-std,mean+std,color=color,alpha=.13)
    ax.set(xlabel='Cumulative net sensing time (min)',ylabel='Global GOSPA (dimensionless)',
           title=f'Fixed urban scene; {len(seeds)} paired repeat'+('s' if len(seeds)>1 else ''))
    ax.grid(alpha=.2);ax.legend(frameon=False);fig.savefig(output/'learning_curve.png',dpi=220)
    if is_main:fig.savefig(output/'learning_curve.pdf')
    plt.close(fig)
    fig=plt.figure(figsize=(10,4.6),constrained_layout=True)
    for i,policy in enumerate(['independent','proposed']):
        ax=fig.add_subplot(1,2,i+1,projection='3d');trajectory=rows(runroot/str(seeds[0])/policy/'trajectory.csv')
        p=np.array([[float(r[k]) for k in ['uav_x','uav_y','uav_z','ugv_x','ugv_y','ugv_z']] for r in trajectory])
        for j,color,label in [(0,'#3972a6','UAV'),(3,'#c65136','UGV')]:ax.plot(p[:,j],p[:,j+1],p[:,j+2],lw=.7,color=color,label=label)
        scene=json.loads((runroot/'scene_manifest.json').read_text())
        for b in scene['buildings']:
            xs=[b[0],b[2],b[2],b[0],b[0]];ys=[b[1],b[1],b[3],b[3],b[1]]
            ax.plot(xs,ys,np.full(5,b[4]),color='#888888',alpha=.65,lw=.6)
            for xx,yy in zip(xs[:4],ys[:4]):
                ax.plot([xx,xx],[yy,yy],[0,b[4]],color='#aaaaaa',alpha=.35,lw=.45)
        ax.plot([0,0],[0,0],[1.5,80],color='#38875a',ls='--',lw=1.2,label='Service (no probes)')
        ax.set(xlim=(0,240),ylim=(0,240),zlim=(0,120),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title=f'{policy}; pair {seeds[0]}')
        ax.set_box_aspect((1,1,.5));ax.view_init(30,-55);ax.legend(loc='upper left',frameon=False,fontsize=8)
    fig.savefig(output/'trajectories.png',dpi=220)
    if is_main:fig.savefig(output/'trajectories.pdf')
    plt.close(fig)
    timing=json.loads((runroot/'campaign_timing.json').read_text())
    stage={}
    for file in runroot.glob('*/*/timings.csv'):
        for row in rows(file):
            for key,value in row.items():
                if key!='round':stage[key]=stage.get(key,0)+float(value)
    rel=np.array([r['relative_improvement'] for r in paired]);absolute=np.array([r['absolute_improvement'] for r in paired])
    text=['# 实际执行报告','',f'运行根目录：`{runroot}`。流程检查：**'+('通过' if flow else '未通过')+'**。',
          f'配对种子：{seeds}；每分支{len(data)//len(checks)}个更新后评价检查点。所有最终数据已进入固定步数训练。',
          '', '| 配对种子 | 独立运动 GOSPA | 所提方法 GOSPA | 相对改善 |','|---|---:|---:|---:|']
    text += [f'| {r["pair_seed"]} | {r["independent_gospa"]:.6f} | {r["proposed_gospa"]:.6f} | {r["relative_improvement"]:.2%} |' for r in paired]
    text += ['',f'逐配对相对改善的平均值为 {rel.mean():.2%}；样本标准差为 {(rel.std(ddof=1) if len(rel)>1 else 0):.2%}。',
             '性能结论：'+('本次固定场景内平均改善为正；三次随机重复不足以支持跨场景泛化或稳定优势的结论。' if absolute.mean()>0 else '本次固定场景未支持平均性能改善，保留全部负或零改善结果。'),
             '',f'主循环实际墙钟：{timing["wall_s"]:.2f} s。射线追踪、参数估计与存储重叠执行，阶段计时不能直接相加作为总墙钟。',
             '共同初始窗口每对只计算一次；分支物理账本各计入该公共窗口。服务下降、驻留和再上升单列，不生成探测记录。',
             '', '计算阶段累计（公共初始化仅计一次）：',*['- '+k+f'：{v:.2f} s' for k,v in stage.items()],
             '', '传播采用 NVIDIA Sionna RT 2.2.0、ITU-R concrete、V 极化、直达和一次镜面反射。双端8元水平UCA由单位范数窄带响应合成。',
             '不包含绕射、漫散射、透射、高阶反射、时钟误差或未知几何。射线真值仅进入信号合成及独立评价；训练标签来自 MUSIC+LS。',
             '方向/时延有限网格、不可分辨多径、频率平滑秩估计、学习拟合及采集域覆盖均可能贡献误差。开发场景结果和门限见 artifacts/calibration/development_diagnostics.json。',
             '两策略共享五次平滑直线原语、安全/速度约束与回收实现；所提方法搜索轴向/垂直/等待候选，基线沿覆盖航点直线移动。动作候选离散方式不同，比较结论针对这两种具体实现。',
             '全局 GOSPA 先加权平均平方集合距离后开方。主曲线为逐种子 GOSPA 的均值和样本标准差；RMSE 表为逐种子 RMSE 均值。',
             '逐种子最终 CSV 保存分项有效匹配数；零有效分母保存 NaN。失败探测与有效空标签独立保存。', '']
    text += ['最终指标（逐种子均值 ± 样本标准差）：','',
             '| 方法 | 全局 GOSPA | GOSPA P90 | 平均漏径 | 平均虚假路径 |',
             '|---|---:|---:|---:|---:|']
    for s in summaries:
        text.append('| '+s['policy']+' | '+' | '.join(f'{s[k+"_mean"]:.6f} ± {s[k+"_std"]:.6f}' for k in
                    ['global_gospa','gospa_p90','mean_miss','mean_false'])+' |')
    text += ['', '| 方法 | 增益 RMSE (dB) | 时延 RMSE (ns) | AoA 方位 (°) | AoA 天顶 (°) | AoD 方位 (°) | AoD 天顶 (°) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        text.append('| '+s['policy']+' | '+' | '.join(f'{s["rmse_"+k+"_mean"]:.4f}' for k in
                    ['gain_db','delay_ns','aoa_az_deg','aoa_zen_deg','aod_az_deg','aod_zen_deg'])+' |')
    service=[]
    for path in sorted(manifests):
        ledger=rows(path.parent/'service_ledger.csv');service.append({'pair_seed':int(path.parent.parent.name),
            'policy':path.parent.name,'net_sensing_s':float(next(r['sensing_time_s'] for r in final if
            int(r['pair_seed'])==int(path.parent.parent.name) and r['policy']==path.parent.name)),
            'service_s':sum(float(r['total_service_s']) for r in ledger),'service_probes':sum(int(r['sensing_probes']) for r in ledger)})
    save_csv(output/'physical_time_ledger.csv',service)
    text += ['', '物理时间见 physical_time_ledger.csv；每分支净测量与服务耗时单列。轨迹图虚线显示独立服务通道，服务不产生探测。','']
    if is_main:
        text += ['完整性审计：runs/main/integrity_audit.json；历史检查点重评价对照：reports/main/reevaluation_validation.json。','']
    (output/'execution_report.md').write_text('\n'.join(text))
    write_json(output/'flow_validation.json',{'passed':flow,'branches':checks,'timing':timing,'stage_totals_s':stage,
             'paired_relative_improvement_mean':float(rel.mean()),'paired_relative_improvement_std':float(rel.std(ddof=1)) if len(rel)>1 else 0})
    print('REPORT',output,'flow_passed',flow,'mean_relative_improvement',rel.mean())
    if is_main and flow:
        from .paper_render import update_manuscript
        update_manuscript(output,summaries,paired,timing)
    return output

def reevaluate(runroot,cfg):
    import torch
    from .geometry import GeometryMap
    from .propagation import SionnaWorld,Evaluator
    from .model import Committee
    g=GeometryMap(cfg);world=SionnaWorld(cfg,g);q,w=g.sample_safe_queries(cfg['queries']['count'],cfg['queries']['seed'])
    e=Evaluator(cfg,world,q,w);features=g.features(q)
    for manifest in Path(runroot).glob('*/*/manifest.json'):
        m=json.loads(manifest.read_text());c=Committee(cfg,m['pair_seed']);data=[]
        for p in sorted((manifest.parent/'checkpoints').glob('round_*.pt'),key=lambda x:int(x.stem.split('_')[-1])):
            state=torch.load(p,map_location=cfg['runtime']['device'],weights_only=False);c.load_state_dict(state['committee'])
            result,errors=e.evaluate(c.predict(features));data.append({'round':state['round'],**result})
        save_csv(manifest.parent/'reevaluated_metrics.csv',data)
    print('Historical checkpoints independently reevaluated')
