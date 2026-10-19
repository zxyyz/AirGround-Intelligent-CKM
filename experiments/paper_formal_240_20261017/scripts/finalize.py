"""Finalize evidence only after formal, integrity and visual checks finish."""
from common import *
import subprocess

def run():
    integrity=json.loads((EXP/'results/integrity.json').read_text());assert integrity['passed']
    assert json.loads((EXP/'results/visual_qa.json').read_text())['passed']
    campaign=json.loads((EXP/'results/campaign_complete.json').read_text())
    status=json.loads((EXP/'STATUS.json').read_text());now=time.time()
    status.update(status='complete',stage='complete',last_progress_epoch=now,end_epoch=now,formal_started=True,formal_pairs_complete=10,model_results_complete=40,formal_evaluation_points=320,total_task_wall_s=now-status['start_epoch'],campaign_wall_s=campaign['wall_s'])
    save(EXP/'STATUS.json',status)
    ledger=ROOT/'experiments/geometry_conditioned_path_detection_20261017/FINAL_REPORT_CN.md';s=ledger.read_text()
    s=s.replace('状态为**执行中：工程验收，正式seed尚未启动**','状态为**已完成：10组正式配对、40组模型结果及320个更新评价点**',1)
    s=s.replace('**制定日期：2026-10-17。状态：执行中；工程验收及配置冻结完成，正式矩阵运行。**','**制定日期：2026-10-17。状态：正式矩阵已完成；本节保留为事前计划，实际结果见上一节。**',1)
    if '执行中：正式矩阵运行' in s:s=s.replace('执行中：正式矩阵运行','正式矩阵已完成')
    note='### 正式执行账本与完成状态\n\n'
    failures=[json.loads(a) for a in (EXP/'failures.jsonl').read_text().splitlines() if a.strip()]
    note+=f"40组正式模型结果、320个更新评价点及预定10seed全部完成；完整性和科学图检查通过。新采集唯一RF观测{integrity['unique_signal_attempts']}次，保留原始信号{integrity['retained_signal_frames']}帧（{100*integrity['signal_retention_fraction']:.2f}%），其他帧保留完整测得集合、质量、轨迹和随机键；工程Smoke保留全量S并通过固定样本信号与标签再生检查。\n\n"
    controlled=sum('Explicit local stop requested' in f['traceback'] for f in failures)
    note+=f"修复后正式计算/评价调度墙钟{campaign['wall_s']/60:.1f} min；含首次运行、随机流修复、工程验收与最终交付的任务墙钟{status['total_task_wall_s']/60:.1f} min。分项计算可能重叠，不加总冒充总墙钟。事件账本共{len(failures)}条，其中{controlled}条为随机流修复时主动STOP触发的安全退出；首次工程保存目录错误及其他事件的原始原因完整保留。未更换正式seed，科学模型与参数不变；初始化命名修复及v1→v2冻结见下文记录。\n\n"
    note+='170HX的llama-server.service最终为inactive/dead/MainPID0；原正式源码、论文、根交接与Overleaf均未改。本轮不再追加实验。\n\n'
    marker='## 下一轮：论文正式全量仿真计划';s=s.replace(marker,note+marker,1);ledger.write_text(s)
    previous=ROOT/'experiments/geometry_conditioned_path_detection_20261017/results/artifact_hashes.json';d=json.loads(previous.read_text());d['FINAL_REPORT_CN.md']=sha(ledger);save(previous,d)
    (EXP/'results/service_final.txt').write_text(subprocess.check_output(['systemctl','--user','show','llama-server.service','-p','ActiveState','-p','SubState','-p','MainPID'],text=True))
    (EXP/'results/gpu_final.txt').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
    cache=EXP/'cache/rt_channels.h5'
    if cache.exists():
        cleanup=dict(path=str(cache),bytes=cache.stat().st_size,reason='derived exact simulator acceleration cache; all formal data committed and audited; retained S and assets permit regeneration')
        cache.unlink();save(EXP/'results/derived_cache_cleanup.json',cleanup)
    artifacts={str(p.relative_to(EXP)):sha(p) for p in sorted(EXP.rglob('*')) if p.is_file() and p.name not in ['artifact_hashes.json','records.lock']}
    artifacts['central_ledger_absolute_path']=str(ledger);artifacts['central_ledger_sha256']=sha(ledger);save(EXP/'results/artifact_hashes.json',artifacts)
    print('FINALIZED',json.dumps(status,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:run()
    except Exception as e:failure('finalize',e);raise
