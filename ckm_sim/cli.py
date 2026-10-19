"""The handoff command interface; every command executes a real implementation."""
import argparse
import json
import subprocess
from pathlib import Path
import yaml
from .config import ROOT,load_config,write_json,canonical_hash,array_hash

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    for command in ['validate','calibrate','run','freeze']:
        p=sub.add_parser(command);p.add_argument('--config',type=Path,required=True)
        if command=='run':
            p.add_argument('--both',action='store_true');p.add_argument('--all-seeds',action='store_true');p.add_argument('--resume',action='store_true')
        if command=='freeze':p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('report');p.add_argument('--run-root',type=Path,required=True)
    p=sub.add_parser('reevaluate');p.add_argument('--config',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='report':
        from .reporting import report
        report(args.run_root);return
    cfg=load_config(args.config)
    if args.command=='validate':
        from tools.validate_handoff_config import validate
        result=validate(args.config,check_main=args.config.stem in ['main','main_frozen'])
        if not result['passed']:raise ValueError(result['errors'])
        from .geometry import GeometryMap
        from .planning import independent_plan,initial_state,sample_trajectory
        g=GeometryMap(cfg);x,w=g.sample_safe_queries(64,192);g.validate_joint_state(x)
        actions,state,log=independent_plan(initial_state(cfg),g,cfg,1800)
        t,positions,velocity=sample_trajectory(actions,cfg);g.validate_joint_state(positions)
        result.update(motion_precheck={'duration_s':1800,'probes':len(t),'all_primitives_safe':all(a.safe(g) for a in actions),'end_state':state,'log':log})
        write_json(ROOT/'artifacts/formula_validation.json',result)
        print('PASS configuration, public geometry and 1800 s motion window',flush=True)
    elif args.command=='calibrate':
        from .calibration import calibrate
        calibrate(cfg)
    elif args.command=='run':
        if cfg['scene']['geometry_seed']==101 and 'frozen_sha256' not in cfg:raise ValueError('Main must be frozen first')
        from .experiment import run
        run(cfg,args.resume)
    elif args.command=='freeze':
        from tools.validate_handoff_config import validate
        from .geometry import GeometryMap
        if not validate(args.config,check_main=True)['passed']:raise ValueError('Main physical locks violated')
        diag=json.loads((ROOT/'artifacts/calibration/development_diagnostics.json').read_text())
        if not (diag['frontend_passed'] and diag['small_sample_fit_passed']):raise ValueError('Development diagnostics incomplete')
        for stage in ['smoke','pilot']:
            for policy in ['independent','proposed']:
                manifests=list((ROOT/'runs'/stage).glob(f'*/{policy}/manifest.json'))
                if not manifests or not all(json.loads(p.read_text())['status']=='complete' for p in manifests):raise ValueError(stage+' incomplete')
        g=GeometryMap(cfg);queries,weights=g.sample_safe_queries(cfg['queries']['count'],cfg['queries']['seed'])
        cfg['freeze']={'scene_geometry_hash':g.hash,'query_hash':array_hash(queries,weights),
                       'git_commit':subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip(),
                       'development_diagnostics_hash':canonical_hash(diag)}
        cfg['estimator']['rank_threshold_value']=diag['calibration']['normalized_rank_threshold']
        calfile=ROOT/'artifacts/calibration'/f'noise_{cfg["scene"]["geometry_seed"]}_{cfg["estimator"]["coarse_delay_bins"]}.json'
        write_json(calfile,diag['calibration'])
        cfg['status']='main_frozen';cfg['frozen_sha256']=canonical_hash(cfg)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(yaml.safe_dump(cfg,sort_keys=False))
        print('FROZEN',args.output,cfg['frozen_sha256'],flush=True)
    elif args.command=='reevaluate':
        from .reporting import reevaluate
        reevaluate(args.run_root,cfg)

if __name__=='__main__':main()
