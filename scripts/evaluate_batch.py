import argparse,json
import torch
from paths import EXP,THRESHOLDS,run_dir
from evaluate import evaluate
p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--variant',required=True);p.add_argument('--pairs',required=True);p.add_argument('--steps',required=True);p.add_argument('--lr',type=float,required=True);p.add_argument('--ratio',type=float,default=1.);p.add_argument('--dataset',default='dev');a=p.parse_args()
torch.set_num_threads(2)
thresholds=THRESHOLDS
if a.dataset=='confirm':
 freeze=json.loads((EXP/'configs/freeze.json').read_text());primary=freeze['thresholds'][a.variant];thresholds=list(dict.fromkeys([primary,.5]))
for pair in map(int,a.pairs.split(',')):
 for step in map(int,a.steps.split(',')):
  evaluate(run_dir(a.stage,a.variant,a.lr,pair,a.ratio)/f'step_{step:06d}',a.dataset,thresholds)
