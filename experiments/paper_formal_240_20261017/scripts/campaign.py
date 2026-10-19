"""Run the predeclared seed queue; scheduling never reads performance values."""
from common import *
import subprocess

def launch(script,args,gpu,log):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    f=(EXP/'logs'/log).open('a')
    p=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(EXP/'scripts'/script),*map(str,args)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    return p,f

def run():
    freeze=json.loads((EXP/'configs/freeze.json').read_text())
    for p,value in freeze['hashes'].items():assert sha(ROOT/p)==value,('frozen input changed',p)
    start=time.perf_counter();todo=list(freeze['seeds']);active=[];completed=[];evaluated=[];evaluation=None
    stage('formal_running')
    while todo or active or evaluation or len(evaluated)<len(completed):
        check_deadline()
        for p,f,pair,gpu in active[:]:
            code=p.poll()
            if code is None:continue
            f.close();active.remove((p,f,pair,gpu))
            if code:raise RuntimeError(('formal worker failed, preserve same seed and checkpoints',pair,code))
            completed.append(pair);print('PAIR_COMPLETE',pair,len(completed),'/10',flush=True)
        if evaluation and evaluation[0].poll() is not None:
            p,f,pair=evaluation;code=p.returncode;f.close();evaluation=None
            if code:raise RuntimeError(('offline evaluator failed',pair,code))
            evaluated.append(pair);print('EVAL_COMPLETE',pair,len(evaluated),'/10',flush=True)
        capacities=json.loads((EXP/'configs/runtime_slots.json').read_text())
        assignment=json.loads((EXP/'configs/device_assignment.json').read_text())
        for pair in todo[:]:
            gpu=assignment[str(pair)]
            if sum(a[3]==gpu for a in active)>=capacities[str(gpu)]:continue
            todo.remove(pair);p,f=launch('runner.py',['--phase','formal','--pair',pair],gpu,f'pair_{pair}.log');active.append((p,f,pair,gpu));print('PAIR_START',pair,'GPU',gpu,flush=True)
        if evaluation is None:
            next_pair=next((pair for pair in completed if pair not in evaluated),None)
            if next_pair is not None:
                p,f=launch('evaluate.py',['--phase','formal','--pair',next_pair],0,f'evaluate_{next_pair}.log');evaluation=p,f,next_pair
        save(EXP/'results/campaign_progress.json',dict(epoch=time.time(),completed_pairs=completed,evaluated_pairs=evaluated,running_pairs=[dict(pair=q,gpu=g,pid=p.pid) for p,_,q,g in active],pending_pairs=todo,elapsed_s=time.perf_counter()-start))
        time.sleep(3)
    save(EXP/'results/campaign_complete.json',dict(status='complete',pairs=completed,evaluated=evaluated,wall_s=time.perf_counter()-start,epoch=time.time()))
    stage('formal_analysis_and_audit');print('FORMAL_CAMPAIGN_COMPLETE',flush=True)

if __name__=='__main__':
    try:run()
    except Exception as e:failure('campaign',e);raise
