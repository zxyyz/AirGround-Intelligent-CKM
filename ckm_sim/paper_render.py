"""Fill the v1.2 manuscript only from persisted frozen experiment results."""
import re
import csv
import numpy as np
from .config import ROOT

def curve_tex(output):
    with (output/'per_seed_round_metrics.csv').open() as f:data=list(csv.DictReader(f))
    result=['\\begin{center}\\begin{tikzpicture}',
        '\\begin{axis}[width=.9\\linewidth,height=50mm,xlabel={累计净测量时间（min）},ylabel={全局GOSPA},',
        'xtick={30,60,90,120,150,180,210,240},grid=major,grid style={gray!20},',
        'tick label style={font=\\scriptsize},label style={font=\\small},legend style={font=\\scriptsize,draw=none},legend pos=north east]']
    for policy,color,label in [('independent','blue!65!black','独立运动'),('proposed','red!70!brown','所提方法')]:
        selected=[r for r in data if r['policy']==policy];seeds=sorted({int(r['pair_seed']) for r in selected})
        matrix=[]
        for seed in seeds:
            curve=sorted([r for r in selected if int(r['pair_seed'])==seed],key=lambda r:int(r['round']))
            t=np.array([float(r['sensing_time_s'])/60 for r in curve]);matrix.append([float(r['global_gospa']) for r in curve])
        matrix=np.array(matrix);mean=matrix.mean(0);std=matrix.std(0,ddof=1)
        def coordinates(v):return ' '.join(f'({x:.0f},{y:.12g})' for x,y in zip(t,v))
        result += ['\\addplot[name path='+policy+'hi,draw=none,forget plot] coordinates {'+coordinates(mean+std)+'};',
                   '\\addplot[name path='+policy+'lo,draw=none,forget plot] coordinates {'+coordinates(mean-std)+'};',
                   '\\addplot[draw=none,fill='+color+',fill opacity=.13,forget plot] fill between[of='+policy+'hi and '+policy+'lo];',
                   '\\addplot[color='+color+',thick,mark=*,mark size=1.3pt] coordinates {'+coordinates(mean)+'};',
                   '\\addlegendentry{'+label+'}']
    result+=['\\end{axis}\\end{tikzpicture}',
             '\\par\\small 图1：固定主场景、三个配对重复；实线为逐种子GOSPA均值，阴影为样本标准差。\\end{center}']
    return '\n'.join(result)+'\n'

def update_manuscript(output,summaries,paired,timing):
    source=(ROOT/'paper/revised/CKM_Active_Cooperative_Measurement_CN_v1_2.tex').read_text()
    destination=ROOT/'paper/final';destination.mkdir(exist_ok=True)
    source=source.replace('中文讨论稿 v1.2','中文讨论稿 v1.3（实际执行）')
    source=source.replace('系统模型、主动建图方法与性能评价','系统模型、主动建图方法与实际配对验证')
    source=source.replace('仿真实现对齐稿','冻结仿真实验结果稿')
    source=source.replace('实验处于方案阶段。','本文已完成一个固定主场景、三组配对随机重复的冻结仿真实验。')
    start=source.index('\\usepackage{newtxmath}');end=source.index('\\IfFontExistsTF{Noto Sans Mono CJK SC}',start)
    source=source[:start]+('\\usepackage{unicode-math}\n\\setmathfont{TeX Gyre Termes Math}\n\\let\\boldsymbol\\symbfit\n'
        '\\let\\ckmwidehat\\widehat\n\\renewcommand{\\widehat}[1]{\\ckmwidehat{#1}}\n'
        '\\let\\ckmwidetilde\\widetilde\n\\renewcommand{\\widetilde}[1]{\\ckmwidetilde{#1}}\n')+source[end:]
    source=source.replace('\\usepackage{booktabs,tabularx,array,microtype}','\\usepackage{booktabs,tabularx,array,microtype,pgfplots}\n\\usepgfplotslibrary{fillbetween}\n\\pgfplotsset{compat=1.18}')
    source=source.replace('\\fontsize{11.2}{15.2}','\\fontsize{10.7}{14.5}')
    source=source.replace('\\fontsize{9}{11.3}', '\\fontsize{8.8}{10.8}')
    source=source.replace('采用短时运动原语与束搜索近似求解，并为候选动作保留可行回收延续。',
        '采用静止端点的五次平滑直线原语与束宽6的搜索近似求解。无人机允许悬停，方向改变时速度和加速度为零，移动段曲率为零；两策略共享原语类型和约束，所提方法的搜索候选限于轴向、垂直及等待，基线沿覆盖航点直线移动。每个动作保留到道路服务点上方80 m的可行回收延续。')
    source=source.replace('服务和模型更新耗时单独记录，无人车位置在轮次间保持连续。',
        '服务和模型更新耗时单独记录，无人车位置在轮次间保持连续。服务点为道路原点，空中会合后沿已核验无建筑碰撞的垂直服务通道下降、驻留60 s并返回80 m；该操作不生成测量记录，单列服务账本。')
    source=source.replace('直达与一次立面反射；固定材料参数生成带噪接收信号','NVIDIA Sionna RT 2.2.0的直达与一次立面镜面反射；ITU-R混凝土和V极化产生Fresnel系数及绝对载波相位，再生成带噪接收信号')
    source=source.replace('\\section{独立运动基线与仿真验证方案}','\\section{独立运动基线与冻结仿真实验}')
    source=source.replace('在水平分区、安全高度层和道路弧长采样点上生成有限联合候选，并按当前轮次的可达性筛选，',
        '在当前轮次保守可达的安全体积与裁剪道路弧长上均匀拒绝采样，采用等权Monte Carlo积分，')
    source=source.replace('候选权重$\\omega_c$由对应空间单元设置。','候选权重$\\omega_c=1/|\\CC|$。')
    source=source.replace('本文提出的组合模型及验证协议依据当前讨论确定，', '本文按冻结的组合模型及验证协议执行，')
    source=source.replace('$M=3$作为初始实验配置。', '$M=3$在全部正式分支中固定。')
    source=source.replace('正式配置见下表。', '正式物理配置沿用该设定。')
    source=source.replace('在线过程由历史测量及几何条件产生决策。',
        '在线过程由历史测量及几何条件产生决策；式\\eqref{eq:problem}由式\\eqref{eq:uncertainty}--\\eqref{eq:planning}的经验代理近似优化，不保证全局最优。')
    begin=source.index('\\begin{center}\n\\small\n\\begin{tabularx}{\\linewidth}{@{}p{29mm}XX@{}}')
    end=source.index('\\subsection{接收信号生成与评价协议}',begin)
    source=source[:begin]+'两方法共享接收机、集合网络、初始化权重与优化器、medoid输出及评价，区别限定为后续联合位置轨迹。\n\n'+source[end:]
    begin=source.index('\\subsection{正式配置与开发校准}')
    end=source.index('每个分支包含最多14400次探测',begin)
    source=source[:begin]+'\\subsection{正式配置与开发校准}\n正式测量采用8窗口$\\times$1800 s，1 Hz。基础/上限高度30/120 m，屋顶净空10 m、水平影响半径30 m；UAV/UGV速度上限2/1 m/s。射频3.5 GHz，双端8元水平UCA，128频点、$\\Delta f=500$ kHz、$Q=64$；信号/前端/CKM路径上限为6/6/3。\n\n'+source[end:]
    source=source.replace('前端门限、训练步数和搜索精度经开发场景校准后统一冻结。',
        '前端以1024次独立噪声校准0.99分位门限，另100次噪声的虚警率为1\\%。Gram矩阵与阶数分解采用complex128，谱计算采用complex64。粗网格为128个时延点、每端12个方位和6个天顶角；保留24个候选、3级$3^5$细化，随后联合LS。三模型各用三层128宽SiLU，AdamW学习率$10^{-3}$；初始化1500步、每轮800步，batch为64。几何候选16个，选测候选256个；$\\ell_U/\\ell_G=40/20$ m、$\\varepsilon=0.1$，在Main前冻结。')
    source=source.replace('输出按统一功率门限、时延范围和上限$K_{\\max}$定义，',
        '输出按$-110$ dB功率门限、$[0,1.8]$ $\\mu$s时延范围和上限$K_{\\max}=3$定义，')
    source=source.replace('所有方法共享探测信号与接收处理。',
        '所有方法共享探测信号与接收处理：$E_p=1$，单位范数阵列下$-90$ dB单路径参考的Frobenius SNR为30 dB，据此固定全场景噪声方差。')
    source=source.replace('分别对应dB、s和rad。',
        '分别对应dB、s和rad，固定为5 dB、25 ns和双端各15度，$c_G=3$。')
    begin=source.index('基线采用无人机分层空域扫描')
    end=source.index('\\subsection{接收信号生成与评价协议}',begin)
    source=source[:begin]+('基线分别执行无人机分层覆盖与无人车道路扫描。两方法共享初始化、前端、网络、预算及运动约束，改变后续联合轨迹；所提方法依据联合位置评分安排移动和等待。\n\n')+source[end:]
    begin=source.index('开发场景用于参数校准，')
    end=source.index('\\subsection{正式配置与开发校准}',begin)
    source=source[:begin]+('开发/主几何种子为9001/101，复用旧高度生成器：240 m城市、16栋建筑、60 m道路\\cite{basedraft}。Sionna RT 2.2.0生成直达及一次立面镜面反射，采用ITU-R混凝土、V极化和绝对载波相位。训练标签只来自带噪接收信号的MUSIC与LS；1024个固定测试查询及射线真值独立管理，三组配对共享初始数据与模型状态。\n\n')+source[end:]
    begin=source.index('正式测量采用8窗口')
    end=source.index('% ---------------- PAGE 6 ----------------',begin)
    source=source[:begin]+('正式测量为8窗口$\\times$1800 s、1 Hz；UAV/UGV速度2/1 m/s、垂直速度0.8 m/s、曲率上限0.05 m$^{-1}$。射频3.5 GHz，双端8元UCA，128频点、500 kHz间隔、$Q=64$；信号/前端/输出路径上限6/6/3。\n\n'
        '独立噪声1024次的0.99分位冻结阶数门限，另100次虚警率为1\\%。Gram分解complex128，谱complex64；粗网格128时延点、双端各12方位和6天顶角，24候选经三级$3^5$细化与联合LS。三模型各用三层128宽SiLU、AdamW学习率$10^{-3}$，初始化1500步、每轮800步、batch 64。几何/选测候选16/256；$\\ell_U/\\ell_G=40/20$ m，$\\varepsilon=0.1$；全部配置在Main前冻结。\n\n')+source[end:]
    begin=source.index('$v_t$为观测有效性标记。')
    end=source.index('\\vspace{3pt}',begin)
    source=source[:begin]+('$v_t$为观测有效性。新增数据进入历史训练；窗口内移动、等待及失败探测计入净测量，服务和计算分列，UGV状态连续。服务点为道路原点上方80 m；经无碰撞垂直通道下降至1.5 m、驻留60 s并返回80 m，该操作不生成探测。\n\n')+source[end:]
    source=source.replace('\\textbf{主图采用“全局重建误差$\\mathcal E_r$—累计净测量时间$t_r$”曲线。}相同时间下比较地图精度，相同误差水平下比较所需测量时间，预算结束时的$\\mathcal E_R$表示最终构建性能。同步报告位置误差分布的高分位数，展示区域内误差差异。',
        '主图按累计净测量时间比较全局误差，最终$\\mathcal E_R$及GOSPA P90列于表中。')
    position=source.index('% ---------------- PAGE 6 ----------------')
    figure=curve_tex(output)
    source=source[:position]+figure+source[position:]
    begin=source.index('\\begin{center}\n\\small\n\\begin{tabularx}{\\linewidth}{@{}p{28mm}p{54mm}X@{}}')
    end=source.index('\\section{结论}',begin)
    table=['\\begin{center}\\small','\\begin{tabular}{lrrrr}\\toprule',
           '方法 & $\\mathcal E_R$ & GOSPA P90 & 漏径 & 虚假路径\\\\\\midrule']
    labels={'independent':'独立运动','proposed':'所提方法'}
    for s in summaries:
        table.append(f'{labels[s["policy"]]} & ${s["global_gospa_mean"]:.4f}\\pm{s["global_gospa_std"]:.4f}$ & {s["gospa_p90_mean"]:.4f} & {s["mean_miss_mean"]:.3f} & {s["mean_false_mean"]:.3f}\\\\')
    table+=['\\bottomrule\\end{tabular}\\end{center}','\\begin{center}\\small',
            '\\begin{tabular}{lrrrrrr}\\toprule','方法 & 增益 & 时延 & AoA方位 & AoA天顶 & AoD方位 & AoD天顶\\\\\\midrule']
    keys=['gain_db','delay_ns','aoa_az_deg','aoa_zen_deg','aod_az_deg','aod_zen_deg']
    for s in summaries:
        table.append(labels[s['policy']]+' & '+' & '.join(f'{s["rmse_"+k+"_mean"]:.2f}' if np.isfinite(s['rmse_'+k+'_mean']) else '--' for k in keys)+'\\\\')
    table+=['\\bottomrule\\end{tabular}\\par 增益单位dB，时延ns，角度度；均为逐种子RMSE的平均值。\\end{center}']
    relative=np.array([r['relative_improvement'] for r in paired]);delta=np.array([r['absolute_improvement'] for r in paired])
    evidence='该固定场景内支持平均性能改善' if delta.mean()>0 else '该固定场景内未支持平均性能改善'
    table+=['每个分支完成14400次探测与8个更新后检查点，三对共有81000次唯一观测，全部有效空标签与失败状态均保留。三个配对的最终相对改善依次为'+
            '、'.join(f'{100*r:.2f}\\%' for r in relative)+f'，均值{100*relative.mean():.2f}\\%，样本标准差{100*relative.std(ddof=1):.2f}\\%。',
            f'主循环实际墙钟为{timing["wall_s"]/60:.2f} min，使用170HX执行MUSIC/训练与2080 Ti执行射线追踪并流水传送观测。物理净测量时间、服务时间和计算墙钟分列；阶段计算重叠，不能简单相加。',
            '数值、数据隔离、配对及全部轨迹审计通过；'+evidence+'。原始信号、检查点及逐种子CSV完整存档。',
            '局限为单一静态几何、已知地图及材料模型；未含绕射、漫散射、透射和高阶反射。搜索候选与覆盖航点的离散方式不同；有限谱网格、近分辨极限多径、采集域覆盖和学习误差仍影响结果，不能解释为实地测量或跨场景普遍优势。\n']
    source=source[:begin]+'\n'.join(table)+source[end:]
    source=source.replace('设置独立运动测量基线开展匹配预算验证。',
        f'完成固定主场景三组配对验证，最终相对改善均值为{100*relative.mean():.2f}\\%，样本标准差为{100*relative.std(ddof=1):.2f}\\%，结论限于本次场景与随机重复。')
    source=source.replace('本文按冻结的组合模型及验证协议执行，本文已完成一个固定主场景、三组配对随机重复的冻结仿真实验。',
        '本文已按冻结的组合模型与协议完成一个固定主场景、三组配对随机重复的仿真实验。')
    begin=source.index('本文形成了由同步接收信号',source.index('\\section{结论}'));end=source.index('\\vspace{2pt}',begin)
    source=source[:begin]+'本文完成了从Sionna RT合成观测、MUSIC估计、集合学习到协同采集的闭环。冻结配对实验的最终地图包含最后一轮数据；'+evidence+'，结论限于本次场景和三组随机重复。\n\n'+source[end:]
    if len(re.findall(r'\\label\{eq:',source))!=19:raise ValueError('Equation label count changed')
    target=destination/'CKM_Active_Cooperative_Measurement_CN_v1_3.tex';target.write_text(source)
    (output/'PAPER_IMPLEMENTATION_SYNC.md').write_text(
        '# 正文与实际实现同步\n\n以v1.2为起点生成v1.3，公式（1）—（19）及标签保留。\n\n'
        '- 用户确认NVIDIA Sionna RT 2.2.0；ITU-R concrete/V极化；关闭透射、绕射、漫散射和高阶反射。\n'
        '- Gram秩估计complex128，谱计算complex64；门限和前端配置由开发数据冻结。\n'
        '- 允许悬停的静止端点五次直线原语保证段间速度、加速度连续；两策略共享原语模型与约束，搜索候选和基线航点离散方式不同，限制已写入正文。\n'
        '- 服务通道与净测量空域分列，下降/驻留/上升不生成探测记录。\n'
        '- 仿真方案改为实际已执行数据，加入曲线、最终表、配对改善和计算墙钟，保留单场景/三重复边界。\n'
        '- 数字全部读取最终CSV，不手写性能值。\n')
    print('MANUSCRIPT',target)
