"""Versioned physical-window runner; software feedback never adds service.

Configuration is the complete matrix. All resumptions are local and science
checked. Long-budget forks copy and verify a completed parent without opening
its HDF5 in a write mode.
"""
import argparse
import copy
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
import h5py
import numpy as np
import torch
from .config import ROOT,write_json,load_config,scientific_hash,canonical_hash,array_hash
from .geometry import GeometryMap
from .propagation import SionnaWorld
from .experiment import (worker_initialize,collect_window,training_data,weight_hash,
    append_csv,service_record,utc,run_manifest)
from .model import Committee
from .v14_learning import LearningCommittee
from .v14_protocol import (isolated_path,policy_kind,load_frontend,check_resume,split_steps,file_hash,frontend_identity)
from .v14_planning import plan_segment,independent_segment,initial_state
from .v14_evaluation import PrivateEvaluator,build_private_queries

def branch_config(cfg,branch,pair):
    c=copy.deepcopy(cfg)
    for k in ['frozen_sha256','freeze','branches','parent_root']:c.pop(k,None)
    for group in ['training','model','acquisition','motion']:c[group].update(branch.get(group,{}))
    c['experiment']['policies']=[branch['policy']];c['experiment']['pair_seeds']=[pair]
    c['active_branch']={k:v for k,v in branch.items() if k not in ['training','model']}
    return c

def prefix_identity(cfg):
    c=copy.deepcopy(cfg)
    for k in ['status','frozen_sha256','freeze']:c.pop(k,None)
    for k in ['output_root','rounds_after_initial']:c['experiment'].pop(k,None)
    return canonical_hash(c)

def validate_matrix(cfg):
    root=isolated_path(ROOT/'configs'/cfg['experiment']['output_root'])
    branches=cfg['branches'];names=[b['name'] for b in branches];seeds=cfg['experiment']['pair_seeds']
    if len(names)!=len(set(names)) or len(seeds)!=len(set(seeds)):raise ValueError('Duplicate branch or seed')
    for b in branches:policy_kind(b['policy'])
    e=cfg['experiment'];interval=e.get('feedback_interval_s',e['round_time_s']);dt=cfg['motion']['action_duration_s']
    if e['round_time_s']%interval or interval%dt:raise ValueError('Feedback must divide physical window into whole actions')
    matrix=[{'pair_seed':s,**b} for s in seeds for b in branches]
    root.mkdir(parents=True,exist_ok=True);path=root/'execution_matrix.json'
    value={'scientific_hash':scientific_hash(cfg),'matrix':matrix,'feedback_interval_s':interval}
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Execution matrix/configuration mismatch')
    if not path.exists():write_json(path,value)
    print('ACTUAL_MATRIX',json.dumps(value),flush=True);return root

def save_checkpoint(path,committee,state,cfg,window,part,update,n,metrics_count,service_count):
    value={'committee':committee.state_dict(),'state':copy.deepcopy(state),'round':window,'part':part,'update_id':update,
        'n_attempts':n,'scientific_config_hash':scientific_hash(cfg),'prefix_identity':prefix_identity(cfg),
        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'weight_hash':weight_hash(committee),
        'metrics_count':metrics_count,'service_count':service_count}
    p=path/'checkpoints';p.mkdir(exist_ok=True);f=p/f'update_{update:04d}.pt';temp=f.with_suffix('.tmp')
    torch.save(value,temp);temp.replace(f);return f

def restore_checkpoint(committee,saved,cfg):
    check_resume(saved,cfg);committee.load_state_dict(saved['committee']);torch.set_rng_state(saved['torch_rng'].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']]);return copy.deepcopy(saved['state'])

def truncate_local(path,saved):
    isolated_path(path)
    with h5py.File(path/'records.h5','a') as h:
        for k in h:h[k].resize(saved['n_attempts'],axis=0)
    for file,n in [('round_metrics.csv',saved['metrics_count']),('timings.csv',saved['metrics_count']),
                   ('trajectory.csv',saved['n_attempts']),('service_ledger.csv',saved['service_count'])]:
        f=path/file
        if f.exists():
            with f.open() as stream:rows=list(csv.DictReader(stream))[:n]
            f.unlink();append_csv(f,rows)
    f=path/'subset_metrics.csv'
    if f.exists():
        with f.open() as stream:rows=[r for r in csv.DictReader(stream) if int(r['update_id'])<=saved['update_id']]
        f.unlink();append_csv(f,rows)

def fork_completed(parent,path,cfg):
    isolated_path(path);manifest=json.loads((parent/'manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError('Only completed parents can be forked')
    if prefix_identity(manifest['config'])!=prefix_identity(cfg):raise ValueError('Parent is not a science-compatible prefix')
    oldwindows=manifest['config']['experiment']['rounds_after_initial'];newwindows=cfg['experiment']['rounds_after_initial']
    if newwindows<=oldwindows:raise ValueError('Fork must extend budget')
    files={str(f.relative_to(parent)):file_hash(f) for f in parent.rglob('*') if f.is_file()}
    shutil.copytree(parent,path)
    if not all(file_hash(path/f)==h for f,h in files.items()):raise ValueError('Fork copy failed hash verification')
    checkpoint=max((path/'checkpoints').glob('update_*.pt'));saved=torch.load(checkpoint,map_location=cfg['runtime']['device'],weights_only=False)
    if saved['prefix_identity']!=prefix_identity(cfg):raise ValueError('Checkpoint prefix differs')
    saved['scientific_config_hash']=scientific_hash(cfg);torch.save(saved,checkpoint)
    provenance={'parent':str(parent),'parent_manifest_sha256':files['manifest.json'],
        'parent_checkpoint_sha256':files[str(checkpoint.relative_to(path))],'parent_files':files,
        'verified_copy':True,'parent_n_attempts':saved['n_attempts'],'prefix_rng_and_optimizers_copied':True}
    write_json(path/'fork_provenance.json',provenance)
    manifest.update(status='running',scientific_config_hash=scientific_hash(cfg),config=cfg,fork=provenance,
        parent_end_utc=manifest.get('end_utc'),start_utc=utc())
    write_json(path/'manifest.json',manifest);return saved,manifest

def train_evaluate(path,committee,state,window,part,update,steps,timings,evaluator,qfeatures,cfg,pair,name,plan):
    features,labels,history,n=training_data(path);before=weight_hash(committee);start=time.perf_counter()
    parts=int(cfg['experiment']['round_time_s']/cfg['experiment'].get('feedback_interval_s',cfg['experiment']['round_time_s']))
    committee.schedule_total_steps=cfg['training']['initialization_steps'] if window==0 else cfg['training']['update_steps_per_round']
    committee.schedule_step_offset=0 if window==0 else sum(split_steps(cfg['training']['update_steps_per_round'],parts)[:part])
    logs=committee.train(features,labels,steps);timings['training_s']=time.perf_counter()-start
    start=time.perf_counter();prediction=committee.predict(qfeatures);timings['prediction_s']=time.perf_counter()-start
    start=time.perf_counter();result,subsets,detail=evaluator.evaluate(prediction);timings['evaluation_s']=time.perf_counter()-start
    sensing=n/cfg['experiment']['probe_frequency_hz']
    row={'policy':name,'pair_seed':pair,'round':window,'part':part,'update_id':update,'sensing_time_s':sensing,
        'n_attempts':n,'n_valid':len(labels),'n_train':len(labels),'steps_per_model':committee.steps[0],**result}
    row.update({k:float(timings.get(k,0)) for k in ['ray_trace_s','rf_estimation_s','features_s','collection_wall_s','training_s','prediction_s','evaluation_s','planning_s']})
    append_csv(path/'round_metrics.csv',[row]);append_csv(path/'timings.csv',[{'update_id':update,**timings}])
    append_csv(path/'subset_metrics.csv',[{'pair_seed':pair,'policy':name,'update_id':update,'sensing_time_s':sensing,**s} for s in subsets])
    np.savez_compressed(path/f'evaluation_update_{update:04d}.npz',**detail)
    write_json(path/f'training_update_{update:04d}.json',{'weight_hash_before':before,'weight_hash_after':weight_hash(committee),
        'models':logs,'plan':plan,'last_observation_used':n-1,'training_data_prefix_attempts':n})
    print(f'METRIC pair={pair} branch={name} window={window} part={part} update={update} probes={n} steps={committee.steps[0]} gospa={result["global_gospa"]:.6f}',flush=True)

def run_campaign(cfg,resume=False,parent_root=None,only_pairs=None):
    root=validate_matrix(cfg);g=GeometryMap(cfg);estimator,cal=load_frontend(cfg)
    qpath=ROOT/'runs/v1_4/private_evaluation'/f'{g.hash}_{cfg["queries"]["seed"]}_{cfg["queries"]["count"]}.npz'
    qpath.parent.mkdir(parents=True,exist_ok=True)
    world=SionnaWorld(cfg,g)
    if qpath.exists():evaluator=PrivateEvaluator(qpath,cfg)
    else:evaluator=build_private_queries(qpath,cfg,g,world)
    if cfg.get('freeze'):
        if cfg['freeze']['geometry_hash']!=g.hash or cfg['freeze']['query_hash']!=evaluator.query_hash:raise ValueError('Frozen geometry/query mismatch')
        for f,h in cfg['freeze']['source_hashes'].items():
            if file_hash(ROOT/f)!=h:raise ValueError('Frozen source mismatch: '+f)
    qfeatures=g.features(evaluator.queries)
    if not (root/'scene_manifest.json').exists():write_json(root/'scene_manifest.json',{'geometry_hash':g.hash,
        'buildings':g.buildings,'query_hash':evaluator.query_hash,'private_evaluator_path':str(qpath),'calibration':cal})
    started=time.perf_counter();e=cfg['experiment'];parts=int(e['round_time_s']/e.get('feedback_interval_s',e['round_time_s']))
    selected=e['pair_seeds'] if only_pairs is None else only_pairs
    if not set(selected)<=set(e['pair_seeds']):raise ValueError('Unregistered dispatch seed')
    dispatch_tag='_'.join(map(str,selected));write_json(root/f'dispatch_{dispatch_tag}.json',{'seeds':selected,'matrix_hash':scientific_hash(cfg),'start_utc':utc()})
    with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn'),initializer=worker_initialize,initargs=(cfg,)) as pool:
        for pair in selected:
            for branch in cfg['branches']:
                branch_started=time.perf_counter()
                name=branch['name'];policy=branch['policy'];c=branch_config(cfg,branch,pair);path=root/str(pair)/name
                cls=Committee if branch.get('learner','L0')=='L0' else LearningCommittee
                committee=cls(c,pair);state=initial_state(c);window=part=update=0
                manifest_path=path/'manifest.json';saved=None
                if manifest_path.exists():
                    if not resume:raise FileExistsError('Existing run: use --resume')
                    manifest=json.loads(manifest_path.read_text());check_resume(manifest,c)
                    if manifest['status']=='complete':continue
                    ckpts=sorted((path/'checkpoints').glob('update_*.pt'))
                    if ckpts:saved=torch.load(ckpts[-1],map_location=c['runtime']['device'],weights_only=False)
                    if saved:state=restore_checkpoint(committee,saved,c);truncate_local(path,saved)
                    else:raise ValueError('Interrupted branch without checkpoint: preserve and restart explicitly')
                elif parent_root:
                    saved,manifest=fork_completed(Path(parent_root)/str(pair)/name,path,c);state=restore_checkpoint(committee,saved,c)
                else:
                    path.mkdir(parents=True);manifest=run_manifest(c,g,world,evaluator.queries,evaluator.weights,pair,name)
                    write_json(manifest_path,manifest)
                try:
                    if saved:
                        window,part,update=saved['round'],saved['part'],saved['update_id']
                    else:
                        # Initialization keyed by learner configuration, not policy or feedback rate.
                        key=canonical_hash({'training':c['training'],'model':c['model'],'pair':pair,'scene':c['scene'],
                            'frontend':frontend_identity(c),'initial_sensing_time_s':e['initial_sensing_time_s'],
                            'probe_frequency_hz':e['probe_frequency_hz'],'metric':c['metric'],'motion':c['motion'],
                            'query_hash':evaluator.query_hash})[:20]
                        common=ROOT/'runs/v1_4/common'/str(pair)/key
                        ready=common/'checkpoints/update_0000.pt'
                        if ready.exists():
                            manifest['common_initial_compute_reused']=True
                            initial=torch.load(ready,map_location=c['runtime']['device'],weights_only=False)
                            committee.load_state_dict(initial['committee']);state=copy.deepcopy(initial['state'])
                            for p in common.iterdir():
                                if p.is_file():shutil.copy2(p,path/p.name)
                            # Common metrics are aliases of a single initialization; rename branch fields.
                            for fn in ['round_metrics.csv','subset_metrics.csv']:
                                f=path/fn
                                with f.open() as stream:rows=list(csv.DictReader(stream))
                                for row in rows:row['policy']=name
                                f.unlink();append_csv(f,rows)
                        else:
                            manifest['common_initial_compute_reused']=False
                            actions,state,plan=independent_segment(state,g,c,e['initial_sensing_time_s'],e['initial_sensing_time_s'])
                            timing=collect_window(path,actions,g,estimator,pool,c,pair,0,'common',0,update_id=0);timing['planning_s']=0.
                            train_evaluate(path,committee,state,0,0,0,c['training']['initialization_steps'],timing,evaluator,qfeatures,c,pair,name,plan)
                            service_record(path,0,state,c,g)
                            common.mkdir(parents=True,exist_ok=True)
                            for p in path.iterdir():
                                if p.is_file() and p.name!='manifest.json':shutil.copy2(p,common/p.name)
                            save_checkpoint(common,committee,state,c,0,0,0,len(training_data(path)[2]),1,1)
                        initial_hash=weight_hash(committee)
                        manifest.update(common_initial_weights_hash=initial_hash,common_initial_records_sha256=file_hash(path/'records.h5'))
                        save_checkpoint(path,committee,state,c,0,0,0,len(training_data(path)[2]),1,1)
                    first_window=window+1 if window==0 or part==parts-1 else window
                    first_part=0 if first_window>window else part+1
                    for physical in range(first_window,e['rounds_after_initial']+1):
                        for sub in range(first_part if physical==first_window else 0,parts):
                            interval=e['round_time_s']/parts;remain=e['round_time_s']-sub*interval
                            history=training_data(path)[2];start=time.perf_counter()
                            # Same per-physical-window seed at its first update as historical policy.
                            seed=pair*10000+physical+(sub*1000000007)
                            actions,state,plan,candidates=plan_segment(policy,committee,g,history,state,c,seed,remain,interval)
                            planning=time.perf_counter()-start;update+=1
                            if candidates is not None:np.savez_compressed(path/f'candidate_update_{update:04d}.npz',**candidates)
                            offset=e['initial_sensing_time_s']+(physical-1)*e['round_time_s']+sub*interval
                            timing=collect_window(path,actions,g,estimator,pool,c,pair,physical,policy,offset,
                                probe_offset=round(sub*interval*e['probe_frequency_hz']),update_id=update)
                            timing['planning_s']=planning
                            steps=split_steps(c['training']['update_steps_per_round'],parts)[sub]
                            train_evaluate(path,committee,state,physical,sub,update,steps,timing,evaluator,qfeatures,c,pair,name,plan)
                            if sub==parts-1:
                                if not np.allclose(state['x'],initial_state(c)['x']):raise ValueError('Service deadline missed')
                                service_record(path,physical,state,c,g)
                            n=len(training_data(path)[2]);services=physical+1 if sub==parts-1 else physical
                            save_checkpoint(path,committee,state,c,physical,sub,update,n,update+1,services)
                            manifest.update(last_completed_window=physical,last_completed_part=sub,last_update=update);write_json(manifest_path,manifest)
                    manifest.update(status='complete',end_utc=utc(),n_attempts=len(training_data(path)[2]),steps_per_model=committee.steps,
                        physical_windows=e['rounds_after_initial']+1,software_updates_after_initial=parts*e['rounds_after_initial'],
                        execution_wall_s=time.perf_counter()-branch_started,torch_peak_memory_bytes=torch.cuda.max_memory_allocated())
                    write_json(manifest_path,manifest)
                except Exception as exc:
                    manifest['status']='failed';manifest['failures'].append({'utc':utc(),'error':repr(exc)});write_json(manifest_path,manifest);raise
    write_json(root/f'campaign_timing_{dispatch_tag}.json',{'wall_s':time.perf_counter()-started,'finished_utc':utc()})
    return root

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--parent-root');p.add_argument('--only-pairs',type=int,nargs='+')
    a=p.parse_args();run_campaign(load_config(a.config),a.resume,a.parent_root,a.only_pairs)

if __name__=='__main__':main()
