"""v1.4 run isolation, explicit policies, calibration identity and fork checks."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import ROOT,canonical_hash,scientific_hash,write_json

LEGACY_ROOT=ROOT.parent/'CKM_Active_Cooperative_v1_2'
POLICIES={'independent','proposed','proposed_v14','coverage_joint_v14','disagreement_only_v14'}

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(8<<20):h.update(b)
    return h.hexdigest()

def isolated_path(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT/'runs/v1_4'):
        raise ValueError('v1.4 outputs must be inside this worktree runs/v1_4')
    if path.is_relative_to(LEGACY_ROOT.resolve()):raise ValueError('Legacy input is read-only')
    return path

def policy_kind(name):
    if name not in POLICIES:raise ValueError('Unknown policy: '+name)
    return 'independent' if name=='independent' else 'joint'

def validate_legacy_run(cfg):
    for p in cfg['experiment']['policies']:
        if p not in {'independent','proposed'}:raise ValueError('Use v14 runner for registered v1.4 policies')
    path=isolated_path(ROOT/'configs'/cfg['experiment']['output_root'])
    matrix=[{'pair_seed':s,'policy':p} for s in cfg['experiment']['pair_seeds'] for p in cfg['experiment']['policies']]
    path.mkdir(parents=True,exist_ok=True)
    f=path/'execution_matrix.json';value={'scientific_hash':scientific_hash(cfg),'matrix':matrix}
    if f.exists() and json.loads(f.read_text())!=value:raise ValueError('Execution matrix/configuration mismatch')
    write_json(f,value);print('ACTUAL_MATRIX',json.dumps(matrix),flush=True)

def frontend_identity(cfg,dependency_file=None):
    # Geometry affects channels, not independent-noise calibration.
    # Include numerical settings, full RF/estimator configuration and dependency lock.
    estimator=copy.deepcopy(cfg['estimator'])
    for k in ['rank_threshold_value','calibration_file','quality_threshold_file']:estimator.pop(k,None)
    return canonical_hash({'sounding':cfg['sounding'],'propagation':cfg['propagation'],
        'estimator':estimator,'ckm_target':cfg['ckm_target'],
        'numerics':{k:cfg['runtime'].get(k) for k in ['rf_dtype','reference_rf_dtype','metric_dtype']},
        'dependencies':file_hash(dependency_file or ROOT/'requirements.scientific.lock')})

def load_frontend(cfg):
    from .config import load_config
    from .music import RFEstimator
    key=frontend_identity(cfg);path=ROOT/'artifacts/v1_4/calibration'/f'{key}.json'
    estimator=RFEstimator(cfg)
    if path.exists():cal=json.loads(path.read_text())
    else:
        baseline=load_config(LEGACY_ROOT/'artifacts/main_frozen.yaml')
        if frontend_identity(baseline,LEGACY_ROOT/'requirements.scientific.lock')==key:
            source=LEGACY_ROOT/'artifacts/calibration/noise_101_128.json'
            cal=json.loads(source.read_text());cal.update(frontend_identity=key,source=str(source),source_sha256=file_hash(source))
            write_json(path,cal)
        else:
            cal=estimator.calibrate(path);cal['frontend_identity']=key;write_json(path,cal)
    if cal.get('frontend_identity')!=key:raise ValueError('Incompatible calibration cache')
    value=cfg['estimator'].get('rank_threshold_value')
    if value is not None and not np.isclose(value,cal['normalized_rank_threshold'],rtol=0,atol=1e-10):
        raise ValueError('Frozen rank threshold and calibration disagree')
    estimator.threshold=cal['normalized_rank_threshold'];return estimator,cal

def check_resume(saved,cfg):
    if saved['scientific_config_hash']!=scientific_hash(cfg):raise ValueError('Resume science hash mismatch')

def split_steps(total,parts):
    q,r=divmod(total,parts);return [q+(i<r) for i in range(parts)]

def heldout_blocks(x):
    x=np.atleast_2d(x)
    return ((np.minimum((x[:,0]/60).astype(int),3)+np.minimum((x[:,1]/60).astype(int),3))%4)==0
