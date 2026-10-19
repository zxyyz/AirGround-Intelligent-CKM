"""Spatially held-out fixed-data prefix replay. No new RF measurements."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from scipy.stats import spearmanr
from .config import ROOT,load_config,write_json,canonical_hash,array_hash
from .geometry import GeometryMap
from .propagation import SionnaWorld
from .experiment import append_csv,weight_hash
from .model import committee_distances,ensemble_medoid
from .v14_learning import LearningCommittee
from .v14_evaluation import PrivateEvaluator,build_private_queries
from .v14_protocol import heldout_blocks,file_hash

def data_prefix(source,n,holdout=True):
    with h5py.File(source/'records.h5','r') as f:
        x=f['x'][:n];mask=f['valid'][:n]
        if holdout:mask &= ~heldout_blocks(x)
        features=f['features'][:n][mask];labels=f['labels'][:n][mask];lengths=f['n_paths'][:n][mask]
        ids=f['record_id'][:n][mask].astype(str)
    return features,[z[:k] for z,k in zip(labels,lengths)],x[mask],ids

def ensure_queries(cfg):
    c=copy.deepcopy(cfg);c['queries'].update(seed=94011,count=1024)
    g=GeometryMap(c);path=ROOT/'runs/v1_4/private_evaluation/development_spatial_holdout.npz'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():ev=PrivateEvaluator(path,c)
    else:ev=build_private_queries(path,c,g,SionnaWorld(c,g),True)
    if not heldout_blocks(ev.queries).all():raise ValueError('Development queries violate spatial holdout')
    return ev,g.features(ev.queries)

def run_replay(setting,multiplier=1,schedule='constant',seeds=(9301,9302,9303),variant='direct'):
    cfg=load_config(ROOT/'configs/v1_4/development/capture.yaml')
    cfg['training'].update(initialization_steps=1500*multiplier,update_steps_per_round=800*multiplier,lr_schedule=schedule)
    cfg['model']['variant']=variant
    ev,qfeatures=ensure_queries(cfg);out=ROOT/'runs/v1_4/development/fixed_data'/setting;out.mkdir(parents=True,exist_ok=True)
    settings={'setting':setting,'multiplier':multiplier,'lr_schedule':schedule,'variant':variant,'spatial_holdout':True,'query_hash':ev.query_hash}
    configfile=out/'setting.json'
    with (out/'setting.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if configfile.exists() and json.loads(configfile.read_text())!=settings:raise ValueError('Replay settings mismatch')
        write_json(configfile,settings)
    start=time.perf_counter()
    for pair in seeds:
        initial=None
        for source_name in ['independent','proposed']:
            source=ROOT/'runs/v1_4/development/capture'/str(pair)/source_name
            manifest=source/'manifest.json'
            if not manifest.exists() or json.loads(manifest.read_text())['status']!='complete':raise ValueError(f'Incomplete fixed data source {source}')
            path=out/str(pair)/source_name;done=path/'complete.json'
            if done.exists():continue
            path.mkdir(parents=True,exist_ok=True);committee=LearningCommittee(cfg,pair)
            sourcehash=file_hash(source/'records.h5');last=0
            ckpts=sorted(path.glob('round_*.pt'))
            if ckpts:
                saved=torch.load(ckpts[-1],map_location=cfg['runtime']['device'],weights_only=False)
                if saved['source_sha256']!=sourcehash:raise ValueError('Source data changed')
                committee.load_state_dict(saved['committee']);last=saved['round']+1
            for round_ in range(last,8):
                n=(round_+1)*1800;features,labels,x,ids=data_prefix(source,n)
                if np.any(heldout_blocks(x)):raise ValueError('Heldout training block leakage')
                steps=cfg['training']['initialization_steps'] if round_==0 else cfg['training']['update_steps_per_round']
                before=weight_hash(committee);tic=time.perf_counter();logs=committee.train(features,labels,steps);wall=time.perf_counter()-tic
                members=committee.predict_members(qfeatures);pred=ensemble_medoid(members,cfg)
                result,subsets,details=ev.evaluate(pred)
                _,u=committee_distances(members,cfg);errors=details['squared_gospa']
                rho=float(spearmanr(u,errors).statistic) if np.ptp(u)>0 and np.ptp(errors)>0 else float('nan')
                counts=np.bincount([len(p) for p in labels],minlength=4)
                row={'setting':setting,'pair_seed':pair,'source':source_name,'round':round_,'sensing_time_s':n,
                    'steps_per_model':committee.steps[0],'n_train':len(labels),'training_wall_s':wall,**result,
                    'mean_training_loss_end':float(np.mean([log['loss_end'] for log in logs])),
                    'disagreement_error_spearman':rho,'zero_disagreement_fraction':float(np.mean(u==0)),
                    'zero_disagreement_error_mean':float(np.mean(errors[u==0])) if np.any(u==0) else float('nan'),
                    'high_disagreement_error_mean':float(np.mean(errors[u>=np.quantile(u,.8)]))}
                append_csv(path/'metrics.csv',[row]);append_csv(path/'subset_metrics.csv',[{k:row[k] for k in ['setting','pair_seed','source','round','sensing_time_s']}|s for s in subsets])
                write_json(path/f'training_{round_}.json',{'models':logs,'weight_before':before,'weight_after':weight_hash(committee),
                    'source_sha256':sourcehash,'record_id_sha256':canonical_hash(ids.tolist()),'prefix_attempts':n,'n_train':len(labels),
                    'label_k_histogram':counts,'unique_rounded_5m_joint_cells':len(np.unique(np.floor(x/5),axis=0)),
                    'time_neighbor_median_joint_distance_m':float(np.median(np.linalg.norm(np.diff(x,axis=0),axis=1)))})
                np.savez_compressed(path/f'evaluation_{round_}.npz',**details,disagreement=u)
                torch.save({'committee':committee.state_dict(),'round':round_,'source_sha256':sourcehash,'settings':settings},path/f'round_{round_}.pt')
                print('REPLAY',setting,pair,source_name,round_,committee.steps[0],result['global_gospa'],flush=True)
            write_json(done,{'source_sha256':sourcehash,'steps':committee.steps,'completed':True})
    write_json(out/('timing_'+'_'.join(map(str,seeds))+'.json'),{'wall_s':time.perf_counter()-start,'seeds':list(seeds)})

def offline_refit(setting,extra_steps,seeds=(9301,9302,9303),tag='offline_refit'):
    cfg=load_config(ROOT/'configs/v1_4/development/capture.yaml');ev,qfeatures=ensure_queries(cfg)
    base=ROOT/'runs/v1_4/development/fixed_data'/setting
    settings=json.loads((base/'setting.json').read_text());cfg['model']['variant']=settings['variant']
    cfg['training'].update(lr_schedule=settings['lr_schedule'],initialization_steps=1500*settings['multiplier'],update_steps_per_round=800*settings['multiplier'])
    for pair in seeds:
        for source_name in ['independent','proposed']:
            path=base/str(pair)/source_name;out=path/tag
            if (out/'complete.json').exists():continue
            out.mkdir(exist_ok=True);saved=torch.load(path/'round_7.pt',map_location=cfg['runtime']['device'],weights_only=False)
            committee=LearningCommittee(cfg,pair);committee.load_state_dict(saved['committee'])
            data=ROOT/'runs/v1_4/development/capture'/str(pair)/source_name
            features,labels,x,ids=data_prefix(data,14400);start=time.perf_counter();logs=committee.train(features,labels,extra_steps)
            training_wall=time.perf_counter()-start;start=time.perf_counter();prediction=committee.predict(qfeatures)
            prediction_wall=time.perf_counter()-start;start=time.perf_counter();result,subsets,details=ev.evaluate(prediction)
            evaluation_wall=time.perf_counter()-start
            write_json(out/'complete.json',{'axis':'extra_optimizer_steps_on_fixed_final_data','extra_steps_per_model':extra_steps,
                'measurement_added':0,'training_wall_s':training_wall,'prediction_wall_s':prediction_wall,
                'evaluation_wall_s':evaluation_wall,**result,'subsets':subsets,'training':logs})
            np.savez_compressed(out/'evaluation.npz',**details);torch.save(committee.state_dict(),out/'committee.pt')

def main():
    p=argparse.ArgumentParser();p.add_argument('--setting',required=True);p.add_argument('--multiplier',type=int,default=1)
    p.add_argument('--schedule',default='constant');p.add_argument('--seeds',type=int,nargs='+',default=[9301,9302,9303]);p.add_argument('--offline-steps',type=int)
    p.add_argument('--variant',default='direct')
    p.add_argument('--offline-tag',default='offline_refit')
    a=p.parse_args()
    if a.offline_steps:offline_refit(a.setting,a.offline_steps,a.seeds,a.offline_tag)
    else:run_replay(a.setting,a.multiplier,a.schedule,a.seeds,a.variant)

if __name__=='__main__':main()
