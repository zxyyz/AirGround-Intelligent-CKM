"""Independent scalar replay, full aggregation and provenance verification."""
import csv,json,gzip
import numpy as np
from paths import EXP,config,sha,save,stamp
from ckm_sim.metrics import squared_gospa
from evaluate import gospa_batch

def validate():
 tasks=json.loads((EXP/'manifests/formal_jobs.json').read_text());freeze=json.loads((EXP/'configs/freeze.json').read_text());ref=np.load(EXP/'cache/reference_confirm.npz');replayed=0;max_error=0.;rows=0
 with gzip.open(EXP/'results/per_query.csv.gz','wt',newline='') as f:
  writer=csv.writer(f);writer.writerow(['variant','pair','ratio','threshold','query','reference_K','predicted_K','squared_gospa','miss','false','medoid_member'])
  for task in tasks:
   d=EXP/task['path'];tr=json.loads((d/'training.json').read_text());assert sha(d/'checkpoint.pt')==tr['checkpoint_sha256']
   ev=json.loads((d/'evaluation_confirm.json').read_text());score=ev['scores'][0];t=score['threshold'];assert t==freeze['thresholds'][task['variant']]
   a=np.load(d/f'predictions_confirm_{t:.2f}.npz');assert len(a['prediction'])==4096
   assert abs(np.sqrt(a['squared_gospa'].mean())-score['gospa'])<1e-12
   assert abs(a['miss'].mean()-score['miss'])<1e-12 and abs(a['false'].mean()-score['false'])<1e-12
   sq,m=gospa_batch(a['prediction'],a['predicted_K'],ref['reference'],ref['n_paths'])
   np.testing.assert_allclose(sq,a['squared_gospa'],rtol=0,atol=1e-11);np.testing.assert_array_equal(m,a['matching'])
   rng=np.random.default_rng(202610270000+task['pair']);selected=[]
   for k in range(4):
    ids=np.flatnonzero(ref['n_paths']==k);selected.extend(rng.choice(ids,min(20,len(ids)),replace=False).tolist())
   if len(selected)<80:selected.extend(rng.choice(np.setdiff1d(np.arange(4096),selected),80-len(selected),replace=False).tolist())
   for i in selected:
    expected,pairs,miss,false=squared_gospa(a['prediction'][i,:a['predicted_K'][i]],ref['reference'][i,:ref['n_paths'][i]],config())
    error=abs(expected-a['squared_gospa'][i]);max_error=max(max_error,error);assert error<1e-10
    assert [(j,int(k)) for j,k in enumerate(a['matching'][i]) if k>=0]==list(pairs)
    replayed+=1
   for i in range(4096):writer.writerow([task['variant'],task['pair'],task['ratio'],t,i,int(ref['n_paths'][i]),int(a['predicted_K'][i]),float(a['squared_gospa'][i]),int(a['miss'][i]),int(a['false'][i]),int(a['medoid_member'][i])]);rows+=1
 hashes={}
 for path,h in {**freeze['asset_hashes'],**freeze['batch_hashes'],**freeze['training_source_hashes']}.items():assert sha(EXP/path)==h,path;hashes[path]=h
 assert replayed>=10000 and len(tasks)==140 and rows==140*4096
 save(EXP/'results/final_validation.json',dict(passed=True,ensemble_runs=len(tasks),members=420,queries_exported=rows,scalar_replays=replayed,maximum_absolute_error=max_error,frozen_hashes_verified=len(hashes),per_query_sha256=sha(EXP/'results/per_query.csv.gz'),created_at=stamp()))
 print('VALIDATED',len(tasks),rows,replayed,max_error,flush=True)
if __name__=='__main__':validate()
