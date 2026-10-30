from common import *
import argparse
p=argparse.ArgumentParser();p.add_argument('action',choices=['train','evaluate']);p.add_argument('--variant',choices=['GeoPool','GeoDirect'],required=True);p.add_argument('--ratio',type=float,required=True);p.add_argument('--pairs',required=True);a=p.parse_args()
assert a.ratio in [.1,.25,.5]
pairs=list(map(int,a.pairs.split(',')));assert all(5101<=v<=5110 for v in pairs)
f=assert_freeze();lr=f['learning_rates'][a.variant]
import torch
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
if a.action=='train':
 import train
 def outdir(stage,variant,lr,pair,ratio=1.):
  assert stage=='formal' and ratio in [.1,.25,.5] and variant in ['GeoPool','GeoDirect']
  return directory(variant,lr,pair,ratio)
 original_stream=train.stream
 def existing_stream(stage,pair,ratio,steps,pool):
  assert (BASE/f'manifests/batches/{stage}_{pair}_{ratio:.2f}.npz').exists()
  return original_stream(stage,pair,ratio,steps,pool)
 train.run_dir=outdir;train.stream=existing_stream
 train.train('formal',pairs,a.variant,56800,a.ratio,lr)
else:
 from evaluate import evaluate
 for pair in pairs:evaluate(directory(a.variant,lr,pair,a.ratio)/'step_056800','confirm',list(dict.fromkeys([f['thresholds'][a.variant],.5])))
