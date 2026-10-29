"""Complete development, freezing, new reference generation and formal training."""
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from paths import EXP,VARIANTS,LRS,THRESHOLDS,run_dir,save,sha,stamp
PYTHON=sys.executable;SCRIPTS=EXP/'scripts';PILOT=[2026102701,2026102702,2026102703]

def job(device,script,args,name):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(device),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='8')
 log=EXP/f'results/logs/{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('a') as f:
  f.write('\nSTART '+stamp()+' '+repr(args)+'\n');f.flush()
  p=subprocess.run([PYTHON,str(SCRIPTS/script),*map(str,args)],cwd=EXP,env=env,stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:
  with (EXP/'results/failures.jsonl').open('a') as f:f.write(json.dumps(dict(time=stamp(),script=script,args=args,returncode=p.returncode,log=str(log)))+'\n')
  raise RuntimeError(f'{name} failed: {p.returncode}')

def run_queues(queues):
 def worker(device,queue):
  with ThreadPoolExecutor(2) as evaluators:
   pending=[]
   for script,args,name in queue:
    for f in pending:
     if f.done():f.result()
    if script=='evaluate_batch.py':pending.append(evaluators.submit(job,device,script,args,name))
    else:job(device,script,args,name)
   for f in pending:f.result()
 with ThreadPoolExecutor(2) as pool:
  futures=[pool.submit(worker,i,q) for i,q in enumerate(queues)]
  for f in futures:f.result()

def precompute(stage,pairs,steps,ratios):
 from train import stream
 source=np.load(EXP/'cache/pool.npz');p={k:source[k] for k in source.files}
 for ratio in ratios:
  for pair in pairs:stream(stage,pair,ratio,steps,p)

def pilot(steps):
 save(EXP/'STATUS.json',dict(status='running',stage=f'pilot_{steps}',updated_at=stamp()))
 precompute('pilot',PILOT,56800,[1.]);queues=[[],[]]
 snapshots=[7100,14200,28400] if steps==28400 else [56800]
 for i,(variant,lr) in enumerate((v,lr) for v in VARIANTS for lr in LRS):
  args=['--stage','pilot','--variant',variant,'--lr',lr,'--pairs',','.join(map(str,PILOT)),'--steps',steps]
  queues[i%2].append(('train.py',args,f'pilot_train_{variant}_{lr}'))
  queues[i%2].append(('evaluate_batch.py',[*args[:-2],'--steps',','.join(map(str,snapshots))],f'pilot_eval_{variant}_{lr}_{steps}'))
 run_queues(queues)

def selection(steps):
 grid={}
 for v in VARIANTS:
  grid[v]={}
  for lr in LRS:
   runs=[json.loads((run_dir('pilot',v,lr,pair)/f'step_{steps:06d}/evaluation_dev.json').read_text()) for pair in PILOT]
   options=[]
   for i,t in enumerate(THRESHOLDS):
    values=[r['scores'][i]['gospa'] for r in runs];options.append(dict(threshold=t,mean=float(np.mean(values)),values=values))
   grid[v][str(lr)]=min(options,key=lambda r:(r['mean'],abs(r['threshold']-.5),r['threshold']))
 selected={};groups=[['Coord'],['RawMap','GeoPool'],['GeoDirect','Full']]
 for group in groups:
  lr=min(LRS,key=lambda lr:(np.mean([grid[v][str(lr)]['mean'] for v in group]),abs(np.log(lr/.001)),lr))
  for v in group:selected[v]=dict(lr=lr,**grid[v][str(lr)])
 result=dict(steps=steps,selected=selected,grid=grid,groups=groups,created_at=stamp())
 save(EXP/f'results/development_selection_{steps}.json',result);return result

def freeze_protocol(steps,selected):
 import xml.etree.ElementTree as ET
 suites=list(ET.parse(EXP/'results/contract_tests.xml').getroot().iter('testsuite'))
 assert suites and all(int(s.get('errors','0'))==int(s.get('failures','0'))==int(s.get('skipped','0'))==0 for s in suites)
 assert json.loads((EXP/'results/tiny_fit.json').read_text())['passed']
 accelerations=list((EXP/'results').glob('acceleration_*.json'));assert len(accelerations)==2 and all(json.loads(p.read_text())['passed'] for p in accelerations)
 assert not (EXP/'cache/reference_confirm.npz').exists()
 precompute('formal',list(range(5101,5111)),steps,[.1,.25,.5,1.])
 sources={str(p.relative_to(EXP)):sha(p) for directory in ['scripts','support','ckm_sim'] for p in sorted((EXP/directory).glob('*.py')) if p.name not in ['analyze.py','report.py','publish.py','finalize.py']}
 assets={str(p.relative_to(EXP)):sha(p) for directory in ['cache','configs','assets'] for p in sorted((EXP/directory).glob('*')) if p.is_file() and p.name not in ['freeze.json','protocol.json']}
 batches={str(p.relative_to(EXP)):sha(p) for p in sorted((EXP/'manifests/batches').glob('formal*.npz'))}
 cfg=json.loads((EXP/'configs/protocol.json').read_text())
 freeze=dict(version=1,frozen=True,created_at=stamp(),steps=steps,learning_rates={v:x['lr'] for v,x in selected.items()},thresholds={v:x['threshold'] for v,x in selected.items()},training_source_hashes=sources,asset_hashes=assets,batch_hashes=batches,primary_contrasts=cfg['primary_contrasts'],formal_ensemble_runs=140)
 save(EXP/'configs/freeze.json',freeze)
 save(EXP/'results/engineering_acceptance.json',dict(passed=True,tests=sum(int(s.get('tests','0')) for s in suites),tiny_fit=True,acceleration_devices=2,created_at=stamp()))
 print('FROZEN',steps,freeze['learning_rates'],freeze['thresholds'],flush=True)

def reference():
 if (EXP/'cache/reference_confirm.npz').exists():return
 save(EXP/'STATUS.json',dict(status='running',stage='fresh_confirmation_reference',updated_at=stamp()))
 queues=[[('prepare_reference.py',['confirm','--shard',i,'--shards',2],f'reference_gpu{i}')] for i in range(2)]
 run_queues(queues);assert (EXP/'cache/reference_confirm.npz').exists()

def formal():
 freeze=json.loads((EXP/'configs/freeze.json').read_text());steps=freeze['steps'];save(EXP/'STATUS.json',dict(status='running',stage='formal',updated_at=stamp()))
 for p,h in {**freeze['asset_hashes'],**freeze['batch_hashes'],**freeze['training_source_hashes']}.items():assert sha(EXP/p)==h,p
 queues=[[],[]];tasks=[]
 for ratio in [.1,.25,.5,1.]:
  variants=VARIANTS if ratio==1 else ['Coord','RawMap','Full']
  for v in variants:
   lr=freeze['learning_rates'][v]
   for device,pairs in enumerate([list(range(5101,5106)),list(range(5106,5111))]):
    args=['--stage','formal','--variant',v,'--lr',lr,'--pairs',','.join(map(str,pairs)),'--steps',steps,'--ratio',ratio]
    queues[device].append(('train.py',args,f'formal_train_{v}_{ratio}_{pairs[0]}'))
    queues[device].append(('evaluate_batch.py',args+['--dataset','confirm'],f'formal_eval_{v}_{ratio}_{pairs[0]}'))
    for pair in pairs:tasks.append(dict(variant=v,pair=pair,ratio=ratio,lr=lr,steps=steps,path=str((run_dir('formal',v,lr,pair,ratio)/f'step_{steps:06d}').relative_to(EXP))))
 assert len(tasks)==140;save(EXP/'manifests/formal_jobs.json',tasks)
 run_queues(queues)
 for task in tasks:assert (EXP/task['path']/'evaluation_confirm.json').exists()
 save(EXP/'results/formal_complete.json',dict(status='complete',runs=140,members=420,steps_per_member=steps,created_at=stamp()))
 save(EXP/'STATUS.json',dict(status='running',stage='analysis_and_publication',updated_at=stamp()))

def all_stages():
 if not (EXP/'configs/freeze.json').exists():
  pilot(28400);chosen=selection(28400);earlier=selection(14200);improvements={}
  for v,c in chosen['selected'].items():
   old=earlier['grid'][v][str(c['lr'])]['mean'];improvements[v]=(old-c['mean'])/old
  extend=any(x>.01 for x in improvements.values());save(EXP/'results/budget_decision.json',dict(extended=extend,relative_improvements=improvements,rule='any selected variant >1% development improvement; extend all',created_at=stamp()))
  steps=56800 if extend else 28400
  if extend:pilot(56800);chosen=selection(56800)
  freeze_protocol(steps,chosen['selected'])
 reference();formal()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','all','reference','formal']);a=p.parse_args()
 if a.stage=='pilot':pilot(28400)
 else:{'all':all_stages,'reference':reference,'formal':formal}[a.stage]()
