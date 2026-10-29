"""Final training-label diagnostic; separate from fresh-query reference scores."""
import csv,json
import numpy as np
import torch
from paths import EXP,VARIANTS,run_dir,save,stamp
from data import features
from evaluate import raw_prediction,decode_members,metrics

def main():
 torch.set_num_threads(2);freeze=json.loads((EXP/'configs/freeze.json').read_text());source=np.load(EXP/'cache/pool.npz');p={k:source[k] for k in source.files};rows=[]
 for variant in VARIANTS:
  f=features('pool',variant)
  for pair in range(5101,5111):
   rng=np.random.default_rng(202610270000+pair);pool=np.flatnonzero(p['train']&(p['pair']==pair));ix=np.sort(rng.choice(pool,512,replace=False))
   directory=run_dir('formal',variant,freeze['learning_rates'][variant],pair)/f'step_{freeze["steps"]:06d}'
   ck=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True);raw,l,_=raw_prediction(ck,f[p['unique_index'][ix]])
   pred,k,_=decode_members(raw,l,freeze['thresholds'][variant]);score,_=metrics(pred,k,p['labels'][ix],p['n_paths'][ix],np.zeros((len(ix),3),bool))
   rows.append(dict(variant=variant,pair=pair,n=512,gospa=score['gospa'],recall=score['recall'],precision=score['precision'],predicted_empty_fraction=score['predicted_empty_fraction'],observed_empty_fraction=score['reference_empty_fraction'],always_empty_gospa=float(np.sqrt(4.5*p['n_paths'][ix].mean()))))
 with (EXP/'results/training_fit_diagnostic.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 save(EXP/'results/training_fit_diagnostic.json',dict(label_source='512 fixed uniformly selected training frames per flow; noisy observation labels, not independent test reference',rows=rows,created_at=stamp()))
if __name__=='__main__':main()
