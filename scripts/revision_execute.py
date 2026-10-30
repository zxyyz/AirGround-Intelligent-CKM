"""Local-only E3 orchestrator. Confirmation is inaccessible until dev freeze."""
import json,os,sys,subprocess,time,threading
import numpy as np
from paths import EXP,save,stamp

def resource_monitor(stop):
 with (EXP/'results/resource_timeline.jsonl').open('a') as f:
  while not stop.is_set():
   raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True)
   f.write(json.dumps(dict(time=stamp(),gpus=raw.strip().splitlines()))+'\n');f.flush();stop.wait(30)
def call(script,args=(),gpu=1):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='4',TORCHINDUCTOR_CACHE_DIR=str(EXP/f'cache/torchinductor_gpu{gpu}'),TRITON_CACHE_DIR=str(EXP/f'cache/triton_gpu{gpu}'))
 dest=EXP/'logs'/('execute_'+script.replace('/','_')+'.log')
 with dest.open('a') as f:
  f.write('\nSTART '+stamp()+' '+str(list(args))+'\n');f.flush()
  ret=subprocess.run([sys.executable,str(EXP/'scripts'/script),*args],cwd=EXP,env=env,stdout=f,stderr=subprocess.STDOUT)
 if ret.returncode:raise RuntimeError(f'{script} failed with {ret.returncode}; {dest}')
def pytest(args):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
 with (EXP/'logs/contract_tests.log').open('a') as f:r=subprocess.run([sys.executable,'-m','pytest','tests','-q',*args],cwd=EXP,env=env,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError('Contract tests failed; see logs/contract_tests.log')
def main():
 start=time.time();stop=threading.Event();monitor=threading.Thread(target=resource_monitor,args=(stop,),daemon=True);monitor.start()
 try:
  if (EXP/'configs/freeze.json').exists():
   import pipeline
   pipeline.reference();pipeline.formal()
   frozen=json.loads((EXP/'configs/freeze.json').read_text());save(EXP/'results/E3_execution_complete.json',dict(formal_groups=140,members=420,development_groups=45,steps=frozen['steps'],resumed=True,wall_seconds_this_process=time.time()-start,created_at=stamp()))
   return
  pos=np.load(EXP/'cache/pool_positions_frozen.npz');n=int((pos['pair']<=5103).sum());assert n==28668
  files=[EXP/f'cache/measurement_parts/{i:06d}.npz' for i in range(0,n,128)]
  while not all(p.exists() and p.with_suffix('.json').exists() for p in files):time.sleep(10)
  if not (EXP/'cache/development_pool.npz').exists():
   parts=[np.load(p) for p in files];p={k:pos[k] if k=='unique' else pos[k][:n] for k in pos.files}
   for k in ['labels','n_paths','valid']:p[k]=np.concatenate([x[k] for x in parts])[:n]
   assert p['valid'].all();p['train']=p['valid'].copy();np.savez_compressed(EXP/'cache/development_pool.npz',**p)
   save(EXP/'results/development_boundary.json',dict(created_at=stamp(),training_noise_flows=[5101,5102,5103],optimization_flows=[2026102701,2026102702,2026102703],records=n,reference='new 480kHz noiseless development labels',new_confirmation_reference_exists=(EXP/'cache/reference_confirm.npz').exists(),formal_groups_started=0))
  while not (EXP/'cache/reference_dev.npz').exists():time.sleep(5)
  pytest(['-k','not training_split','--junitxml='+str(EXP/'results/partial_contract_tests.xml')]);call('check_acceleration.py');call('check_fit.py')
  save(EXP/'STATUS.json',dict(status='running',stage='E3_development',formal_complete=0,development_planned_groups=45,formal_planned_groups=140,updated_at=stamp()))
  import pipeline
  pipeline.pilot(28400)
  chosen=pipeline.selection(28400);earlier=pipeline.selection(14200)
  improvements={v:(earlier['grid'][v][str(c['lr'])]['mean']-c['mean'])/earlier['grid'][v][str(c['lr'])]['mean'] for v,c in chosen['selected'].items()}
  extend=any(x>.01 for x in improvements.values());save(EXP/'results/budget_decision.json',dict(extended=extend,relative_improvements=improvements,rule='any selected variant >1% dev improvement; extend every candidate and variant',created_at=stamp()))
  steps=56800 if extend else 28400
  if extend:pipeline.pilot(56800);chosen=pipeline.selection(56800)
  for i in range(2):
   while not (EXP/f'results/new_measurement_resources_{i}.json').exists():time.sleep(10)
  call('revision_rebuild.py',['assemble']);pytest(['--junitxml='+str(EXP/'results/contract_tests.xml')])
  pipeline.freeze_protocol(steps,chosen['selected']);pipeline.reference();pipeline.formal()
  save(EXP/'results/E3_execution_complete.json',dict(formal_groups=140,members=420,development_groups=45,steps=steps,wall_seconds=time.time()-start,created_at=stamp()))
 finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':
 try:main()
 except Exception as e:save(EXP/'results/execution_error.json',dict(error=repr(e),time=stamp()));raise
