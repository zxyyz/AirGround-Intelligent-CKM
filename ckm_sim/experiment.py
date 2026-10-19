"""Fixed-budget paired closed loop. All measured frames and resumable state persist."""
import copy
import csv
import hashlib
import json
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import h5py
import numpy as np
import torch
from .config import ROOT,write_json,scientific_hash,array_hash,canonical_hash
from .geometry import GeometryMap
from .propagation import SionnaWorld,SoundingOracle,Evaluator
from .music import RFEstimator
from .model import Committee
from .planning import initial_state,independent_plan,beam_plan,sample_trajectory,budget_ledger
from .acquisition import score_candidates

_WORKER_ORACLE=None
def worker_initialize(cfg):
    global _WORKER_ORACLE
    os.environ['CUDA_VISIBLE_DEVICES']=str(cfg['runtime']['rt_worker_visible_device'])
    geom=GeometryMap(cfg);world=SionnaWorld(cfg,geom);_WORKER_ORACLE=SoundingOracle(cfg,world)

def worker_observe(request):
    positions,pair,round_,indices=request;signals=[];start=time.perf_counter()
    for x,index in zip(positions,indices):signals.append(_WORKER_ORACLE.observe(x,(pair,round_,int(index)))[0])
    return np.stack(signals),time.perf_counter()-start

def utc():return datetime.now(timezone.utc).isoformat()

def weight_hash(committee):
    h=hashlib.sha256()
    for model in committee.models:
        for key,value in model.state_dict().items():h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def append_csv(path,rows):
    rows=list(rows)
    if not rows:return
    path=Path(path);exists=path.exists()
    with path.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        if not exists:writer.writeheader()
        writer.writerows(rows)

def prepare_records(path,cfg):
    nmax=cfg['ckm_target']['max_paths'];s=cfg['sounding']
    h=h5py.File(path,'a')
    spec={'x':((6,),np.float64),'features':((6+10*cfg['model']['geometry_candidate_count'],),np.float32),'labels':((nmax,6),np.float64),
          'n_paths':((),np.int32),'valid':((),bool),'round':((),np.int32),'probe_index':((),np.int32),
          'time_s':((),np.float64),'physical_window_id':((),np.int32),'update_id':((),np.int32),
          'global_probe_index':((),np.int64),'record_id':((),h5py.string_dtype('utf-8')),
          'quality':((),h5py.string_dtype('utf-8')),
          'S':((s['n_frequencies'],s['nr'],s['pilot_count']),np.complex64)}
    for key,(shape,dtype) in spec.items():
        if key not in h:h.create_dataset(key,shape=(0,*shape),maxshape=(None,*shape),dtype=dtype,chunks=(16,*shape))
    h.attrs['scientific_config_hash']=scientific_hash(cfg)
    h.attrs['label_source']='RFEstimator(S,pilots,noise_calibration)'
    h.attrs['raw_signals_retained']=True
    return h

def append_records(h,rows):
    start=len(h['x']);n=len(rows)
    for key in h:
        h[key].resize(start+n,axis=0);h[key][start:]=[r[key] for r in rows]
    h.flush()

def collect_window(path,actions,geometry,estimator,pool,cfg,pair,round_,policy,offset,probe_offset=0,update_id=None):
    t,x,v=sample_trajectory(actions,cfg);geometry.validate_joint_state(x)
    f=cfg['experiment']['probe_frequency_hz'];expected=round(sum(a.duration for a in actions)*f)
    if len(t)!=expected:raise ValueError('Probe ledger mismatch')
    start=time.perf_counter();features=geometry.features(x);feature_time=time.perf_counter()-start
    # Ray tracing on RTX overlaps observation-only MUSIC on the CMP device.
    batch=32;futures=[]
    for i in range(0,len(x),batch):
        indices=np.arange(i,min(i+batch,len(x)));futures.append(pool.submit(worker_observe,(x[indices],pair,round_,indices+probe_offset)))
    h=prepare_records(path/'records.h5',cfg);k=cfg['ckm_target']['max_paths'];rf_time=rt_time=0;valid=0
    pilots=np.eye(cfg['sounding']['nt'])*np.sqrt(cfg['sounding']['pilot_energy_ep'])
    wall=time.perf_counter()
    for bi,future in enumerate(futures):
        signals,rt=future.result();rt_time+=rt;rows=[]
        before=time.perf_counter()
        try:estimated,qualities=estimator.estimate_many(signals,pilots)
        except Exception as exc:
            print('BATCH_RF_FALLBACK',repr(exc),flush=True)
            estimated=[];qualities=[]
            for signal in signals:
                try:z,q=estimator.estimate(signal,pilots)
                except Exception as error:z=np.empty((0,6));q={'valid':False,'reason':repr(error),'order':-1}
                estimated.append(z);qualities.append(q)
        rf_time+=time.perf_counter()-before
        for j,signal in enumerate(signals):
            index=bi*batch+j;labels,quality=estimated[j],qualities[j]
            valid+=quality['valid'];padded=np.zeros((k,6));padded[:len(labels)]=labels
            origin='common' if round_==0 else policy
            rows.append({'x':x[index],'features':features[index],'labels':padded,'n_paths':len(labels),
                'valid':quality['valid'],'round':round_,'probe_index':index+probe_offset,'time_s':offset+t[index],
                'physical_window_id':round_,'update_id':round_ if update_id is None else update_id,
                'global_probe_index':round(offset*f)+index,
                'record_id':f'{pair}:{origin}:{round_}:{index+probe_offset}','quality':json.dumps(quality),'S':signal})
        append_records(h,rows)
        if bi%10==0 or bi==len(futures)-1:
            print(f'PROBES pair={pair} policy={policy} round={round_} {min((bi+1)*batch,len(x))}/{len(x)} valid={valid}',flush=True)
    h.close()
    trajectory=[dict(round=round_,time_s=float(offset+tm),**{name:float(value) for name,value in zip(
                ['uav_x','uav_y','uav_z','ugv_x','ugv_y','ugv_z','uav_vx','uav_vy','uav_vz','ugv_vx','ugv_vy','ugv_vz'],np.r_[position,velocity])})
                for tm,position,velocity in zip(t,x,v)]
    append_csv(path/'trajectory.csv',trajectory)
    suffix='' if update_id is None else f'_update_{update_id}'
    write_json(path/f'trajectory_round_{round_}{suffix}.json',[a.serialize() for a in actions])
    return {'ray_trace_s':rt_time,'rf_estimation_s':rf_time,'features_s':feature_time,
            'collection_wall_s':time.perf_counter()-wall,'n_window_attempts':len(t),'n_window_valid':int(valid)}

def training_data(path):
    with h5py.File(path/'records.h5') as h:
        mask=h['valid'][:];features=h['features'][:][mask];labels=h['labels'][:][mask];lengths=h['n_paths'][:][mask]
        history=h['x'][:];n_attempts=len(history)
    return features,[p[:n] for p,n in zip(labels,lengths)],history,n_attempts

def checkpoint(path,committee,state,round_,n_attempts,cfg):
    path.mkdir(exist_ok=True)
    value={'committee':committee.state_dict(),'state':state,'round':round_,'n_attempts':n_attempts,
           'scientific_config_hash':scientific_hash(cfg),'torch_rng':torch.get_rng_state(),
           'cuda_rng':torch.cuda.get_rng_state_all(),'weight_hash':weight_hash(committee)}
    file=path/f'round_{round_}.pt';temporary=path/f'round_{round_}.tmp';torch.save(value,temporary);temporary.replace(file)
    return value['weight_hash']

def service_record(path,round_,state,cfg,geometry):
    # The safe airborne domain excludes a separately defined landing/service corridor.
    # The vertical corridor at (0,0) is outside every building footprint.
    x=np.array(state['x']);landing=x.copy();landing[2]=cfg['scene']['ugv_antenna_height_m']
    if not geometry.clear(x[:3],landing[:3]):raise ValueError('Service corridor intersects building')
    dz=abs(x[2]-landing[2]);leg=1.875*dz/cfg['motion']['uav_vertical_speed_max_mps'];dwell=60.
    append_csv(path/'service_ledger.csv',[{'round':round_,'descent_s':leg,'service_dwell_s':dwell,
                                        'ascent_s':leg,'total_service_s':2*leg+dwell,'sensing_probes':0}])
    return 2*leg+dwell

def evaluate_round(path,committee,state,round_,timings,evaluator,query_features,cfg,pair,policy,plan_log):
    features,labels,history,n_attempts=training_data(path)
    before=weight_hash(committee);start=time.perf_counter()
    steps=cfg['training']['initialization_steps'] if round_==0 else cfg['training']['update_steps_per_round']
    trainlog=committee.train(features,labels,steps);timings['training_s']=time.perf_counter()-start
    start=time.perf_counter();predictions=committee.predict(query_features);timings['prediction_s']=time.perf_counter()-start
    result,errors=evaluator.evaluate(predictions)
    value=weight_hash(committee)
    count=np.array([len(p) for p in predictions]);counts=np.array([len(p) for p in labels])
    metric={'policy':policy,'pair_seed':pair,'round':round_,'sensing_time_s':budget_ledger(cfg)[round_],
            'n_attempts':n_attempts,'n_valid':len(labels),'n_train':len(labels),**result,
            'prediction_empty_fraction':float(np.mean(count==0)),'label_empty_fraction':float(np.mean(counts==0)),
            'label_at_cap_fraction':float(np.mean(counts==cfg['ckm_target']['max_paths'])),
            **{k:float(timings.get(k,0)) for k in ['ray_trace_s','rf_estimation_s','features_s','collection_wall_s','training_s','prediction_s','planning_s']}}
    append_csv(path/'round_metrics.csv',[metric]);np.savez_compressed(path/f'per_query_metrics_round_{round_}.npz',squared_gospa=errors,path_counts=count)
    write_json(path/f'training_round_{round_}.json',{'weight_hash_before':before,'weight_hash_after':value,'models':trainlog,'plan':plan_log})
    append_csv(path/'timings.csv',[{'round':round_,**{k:float(timings.get(k,0)) for k in ['ray_trace_s','rf_estimation_s','features_s','collection_wall_s','training_s','prediction_s','planning_s']}}])
    checkpoint(path/'checkpoints',committee,state,round_,n_attempts,cfg)
    print(f'METRIC pair={pair} policy={policy} round={round_} gospa={result["global_gospa"]:.6f} train={len(labels)} weight={value[:12]}',flush=True)
    return value

def run_manifest(cfg,g,world,queries,weights,pair,policy):
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip()
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'ckm_sim').glob('*.py'))}
    return {'status':'running','start_utc':utc(),'pair_seed':pair,'policy':policy,'code_commit':head if len(head)==40 else None,
            'source_hashes':source_hashes,
            'scientific_config_hash':scientific_hash(cfg),'scene_geometry_hash':g.hash,'material_hash':world.material_hash,
            'materials':world.material_document,'query_hash':array_hash(queries,weights),
            'source_paper_hash':hashlib.sha256((ROOT/'paper/revised/CKM_Active_Cooperative_Measurement_CN_v1_2.tex').read_bytes()).hexdigest(),
            'noise_pairing':'pair_seed,round,probe_index','runtime':cfg['runtime'],'config':cfg,'versions':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},
            'failures':[]}

def trim_to_checkpoint(path,checkpoint_):
    n=checkpoint_['n_attempts'];round_=checkpoint_['round']
    with h5py.File(path/'records.h5','a') as h:
        for key in h:h[key].resize(n,axis=0)
    for name in ['round_metrics.csv','timings.csv','trajectory.csv','service_ledger.csv']:
        file=path/name
        if not file.exists():continue
        with file.open() as f:rows=list(csv.DictReader(f))
        rows=[r for r in rows if int(r['round'])<=round_];file.unlink();append_csv(file,rows)

def run(cfg,resume=False):
    from .v14_protocol import validate_legacy_run,load_frontend
    validate_legacy_run(cfg)
    g=GeometryMap(cfg);world=SionnaWorld(cfg,g)
    estimator,cal=load_frontend(cfg)
    queries,weights=g.sample_safe_queries(cfg['queries']['count'],cfg['queries']['seed'])
    evaluator=Evaluator(cfg,world,queries,weights);query_features=g.features(queries)
    runroot=(ROOT/'configs'/cfg['experiment']['output_root']).resolve();runroot.mkdir(parents=True,exist_ok=True)
    evaluator.save_private(runroot/'private_frozen_queries.npz')
    write_json(runroot/'scene_manifest.json',{'geometry_hash':g.hash,'buildings':g.buildings,'roads':g.nodes,'edges':g.edges,
        'generator':'legacy make_district PCG64DXSM row-major 16 uniform heights','materials':world.material_document,
        'query_hash':array_hash(queries,weights),'calibration':cal})
    campaign=time.perf_counter()
    with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn'),initializer=worker_initialize,initargs=(cfg,)) as pool:
        for pair in cfg['experiment']['pair_seeds']:
            common=runroot/str(pair)/'common';common.mkdir(parents=True,exist_ok=True)
            existing=common/'checkpoints/round_0.pt';initial=Committee(cfg,pair);state=initial_state(cfg)
            if existing.exists():
                if not resume:raise FileExistsError('Use --resume to preserve existing experiment')
                saved=torch.load(existing,map_location=cfg['runtime']['device'],weights_only=False)
                if saved['scientific_config_hash']!=scientific_hash(cfg):raise ValueError('Resume configuration mismatch')
                initial.load_state_dict(saved['committee']);state=saved['state'];initial_weight=saved['weight_hash']
                trim_to_checkpoint(common,saved)
            else:
                if (common/'records.h5').exists():
                    # Preserve interrupted records instead of overwriting a failed initial window.
                    if not resume:raise FileExistsError('Interrupted common window; use --resume')
                    historydir=common/'interrupted_initial'/utc().replace(':','-');historydir.mkdir(parents=True)
                    for p in list(common.iterdir()):
                        if p.is_file():shutil.move(p,historydir/p.name)
                start=time.perf_counter();actions,state,plan=independent_plan(state,g,cfg,cfg['experiment']['initial_sensing_time_s'])
                planning=time.perf_counter()-start
                timings=collect_window(common,actions,g,estimator,pool,cfg,pair,0,'common',0);timings['planning_s']=planning
                initial_weight=evaluate_round(common,initial,state,0,timings,evaluator,query_features,cfg,pair,'common',plan)
                service_record(common,0,state,cfg,g)
            with h5py.File(common/'records.h5') as h:common_hash=array_hash(h['x'][:],h['labels'][:],h['n_paths'][:],h['S'][:])
            for policy in cfg['experiment']['policies']:
                path=runroot/str(pair)/policy;path.mkdir(parents=True,exist_ok=True);manifest_path=path/'manifest.json'
                manifest=run_manifest(cfg,g,world,queries,weights,pair,policy)
                committee=initial.clone();current=copy.deepcopy(state);completed=0
                if manifest_path.exists():
                    if not resume:raise FileExistsError('Existing branch; use --resume')
                    manifest=json.loads(manifest_path.read_text())
                    if manifest['scientific_config_hash']!=scientific_hash(cfg):raise ValueError('Resume configuration mismatch')
                    checkpoints=sorted((path/'checkpoints').glob('round_*.pt'),key=lambda x:int(x.stem.split('_')[-1]))
                    if checkpoints:
                        saved=torch.load(checkpoints[-1],map_location=cfg['runtime']['device'],weights_only=False)
                        committee.load_state_dict(saved['committee']);current=saved['state'];completed=saved['round']
                        torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in saved['cuda_rng']])
                        trim_to_checkpoint(path,saved)
                    if manifest['status']=='complete':continue
                else:
                    shutil.copyfile(common/'records.h5',path/'records.h5')
                    shutil.copyfile(common/'trajectory.csv',path/'trajectory.csv')
                    shutil.copyfile(common/'service_ledger.csv',path/'service_ledger.csv')
                    with (common/'round_metrics.csv').open() as f:row=next(csv.DictReader(f));row['policy']=policy
                    append_csv(path/'round_metrics.csv',[row])
                    checkpoint(path/'checkpoints',committee,current,0,len(training_data(path)[2]),cfg)
                    manifest.update(common_initial_records_hash=common_hash,common_initial_weights_hash=initial_weight)
                write_json(manifest_path,manifest)
                try:
                    for round_ in range(completed+1,cfg['experiment']['rounds_after_initial']+1):
                        history=training_data(path)[2];start=time.perf_counter();candidate_log={}
                        if policy=='proposed':
                            c,w,j,u,h=score_candidates(committee,g,history,cfg,pair*10000+round_,cfg['experiment']['round_time_s'])
                            if np.any(np.linalg.norm(c[:,None]-queries[None],axis=-1)<cfg['queries']['equal_coordinate_tolerance_m']):raise ValueError('Candidate equals frozen test query')
                            np.savez_compressed(path/f'candidate_round_{round_}.npz',positions=c,weights=w,score=j,disagreement=u,historical_coverage=h)
                            candidate_log={'disagreement_min':float(u.min()),'disagreement_max':float(u.max()),
                                           'min_candidate_query_euclidean_m':float(np.linalg.norm(c[:,None]-queries[None],axis=-1).min())}
                            actions,current,plan=beam_plan(current,g,cfg,cfg['experiment']['round_time_s'],c,w,j)
                        else:actions,current,plan=independent_plan(current,g,cfg,cfg['experiment']['round_time_s'])
                        planning=time.perf_counter()-start;plan.update(candidate_log)
                        timings=collect_window(path,actions,g,estimator,pool,cfg,pair,round_,policy,budget_ledger(cfg)[round_-1]);timings['planning_s']=planning
                        evaluate_round(path,committee,current,round_,timings,evaluator,query_features,cfg,pair,policy,plan)
                        service_record(path,round_,current,cfg,g)
                        manifest['last_completed_round']=round_;write_json(manifest_path,manifest)
                    manifest.update(status='complete',end_utc=utc());write_json(manifest_path,manifest)
                except Exception as exc:
                    failure={'time_utc':utc(),'error':repr(exc),'last_completed_round':manifest.get('last_completed_round',0)}
                    manifest['failures'].append(failure);manifest['status']='failed';write_json(manifest_path,manifest)
                    write_json(path/f'failure_{utc().replace(":","-")}.json',failure);raise
    write_json(runroot/'campaign_timing.json',{'start_seconds':campaign,'wall_s':time.perf_counter()-campaign,'finished_utc':utc()})
    return runroot
