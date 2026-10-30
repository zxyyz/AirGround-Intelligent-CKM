from common import *
import os,sys,time,subprocess,threading,fcntl
from concurrent.futures import ThreadPoolExecutor

status_lock=threading.Lock()

def status():
 with status_lock:return _status()

def _status():
 rows=[]
 for t in tasks():
  p=EXT/t['path'];s='evaluated' if (p/'evaluation_confirm.json').exists() else 'trained' if (p/'checkpoint.pt').exists() else 'pending'
  rows.append({**t,'status':s})
 save(EXT/'manifests/task_status.json',dict(updated_at=stamp(),tasks=rows,completed=sum(r['status']=='evaluated' for r in rows),trained=sum(r['status'] in ['trained','evaluated'] for r in rows),planned=60))
 return rows

def monitor(stop):
 with (EXT/'results/device_timeline.jsonl').open('a') as out:
  while not stop.is_set():
   try:
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True)
    out.write(json.dumps(dict(time=stamp(),devices=raw.strip().splitlines()))+'\n');out.flush();status()
   except Exception as e:print('MONITOR',repr(e),flush=True)
   stop.wait(30)

def worker(gpu):
 pairs=list(range(5101+5*gpu,5106+5*gpu))
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='4',TORCHINDUCTOR_CACHE_DIR=str(EXT/f'cache/torchinductor_gpu{gpu}'),TRITON_CACHE_DIR=str(EXT/f'cache/triton_gpu{gpu}'),MPLCONFIGDIR=str(EXT/'cache/matplotlib'))
 for ratio in [.1,.25,.5]:
  for variant in ['GeoPool','GeoDirect']:
   f=read(BASE/'configs/freeze.json');paths=[directory(variant,f['learning_rates'][variant],p,ratio)/'step_056800' for p in pairs]
   for action in ['train','evaluate']:
    if all((p/('training.json' if action=='train' else 'evaluation_confirm.json')).exists() for p in paths):continue
    if action=='train':
     while True:
      free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
      if free>=1500:break
      print('RESOURCE_WAIT',gpu,free,flush=True);time.sleep(30)
    args=[sys.executable,str(EXT/'scripts/run_group.py'),action,'--variant',variant,'--ratio',str(ratio),'--pairs',','.join(map(str,pairs))]
    start=time.time();log=EXT/f'logs/{action}_{variant}_{ratio:.2f}_gpu{gpu}.log'
    with log.open('a') as out:
     out.write('\nSTART '+stamp()+' '+repr(args)+'\n');out.flush();proc=subprocess.run(args,cwd=EXT,env=env,stdout=out,stderr=subprocess.STDOUT)
    result=dict(action=action,variant=variant,ratio=ratio,pairs=pairs,gpu=gpu,returncode=proc.returncode,wall_seconds=time.time()-start,time=stamp(),log=str(log.relative_to(EXT)))
    with (EXT/f'results/execution_gpu{gpu}.jsonl').open('a') as out:out.write(json.dumps(result)+'\n')
    if proc.returncode:raise RuntimeError(result)
    status()

def main():
 lock=(EXT/'audit/execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert_freeze();save(EXT/'STATUS.json',dict(status='running',stage='supplementary_60_training',started_at=stamp(),planned_groups=60))
 stop=threading.Event();thread=threading.Thread(target=monitor,args=(stop,),daemon=True);thread.start();start=time.time()
 try:
  with ThreadPoolExecutor(2) as pool:
   jobs=[pool.submit(worker,gpu) for gpu in [0,1]]
   for job in jobs:job.result()
  rows=status();assert all(t['status']=='evaluated' for t in rows)
  save(EXT/'results/execution_complete.json',dict(complete=True,groups=60,members=180,updates=180*56800,wall_seconds=time.time()-start,finished_at=stamp()))
  save(EXT/'STATUS.json',dict(status='running',stage='independent_validation_pending',groups_complete=60,updated_at=stamp()))
 finally:stop.set();thread.join(timeout=5)
if __name__=='__main__':
 try:main()
 except Exception as e:save(EXT/'results/execution_error.json',dict(error=repr(e),time=stamp()));raise
