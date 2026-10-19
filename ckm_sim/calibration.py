"""Development diagnostics; no final-query score is used to choose settings."""
import copy
import json
import time
import numpy as np
import torch
from .config import ROOT,write_json,scientific_hash
from .geometry import GeometryMap,directions
from .propagation import SionnaWorld,SoundingOracle
from .music import RFEstimator
from .model import Committee
from .sounding import channel_from_paths,simulate_received
from .metrics import squared_gospa,spherical_angle

def calibrate(cfg):
    out=ROOT/'artifacts/calibration';out.mkdir(exist_ok=True)
    g=GeometryMap(cfg);world=SionnaWorld(cfg,g);oracle=SoundingOracle(cfg,world);rf=RFEstimator(cfg)
    calfile=out/f'noise_{cfg["scene"]["geometry_seed"]}_{cfg["estimator"]["coarse_delay_bins"]}.json'
    cal=rf.calibrate(calfile);controlled=[]
    examples=[np.array([[-80,rf.delay[20],*rf.rxangles[9],*rf.txangles[22]]]),
              np.array([[-80,300e-9,.3,.7,-1,2.4],[-85,700e-9,2.2,1.1,1,2.]]),
              np.array([[-80,300e-9,.3,.7,-1,2.4],[-85,700e-9,2.2,1.1,1,2.],[-90,1200e-9,-2,1.3,2.6,2.8]])]
    for i,z in enumerate(examples):
        h=channel_from_paths(z,10**(z[:,0]/20)*np.exp(1j*np.arange(len(z))),cfg)
        s,p=simulate_received(h,(9001,900,i),cfg,noiseless=i==0)
        estimated,q=rf.estimate(s,p);d,pairs,miss,false=squared_gospa(estimated,z,cfg)
        error=[]
        for a,b in pairs:
            error.append([abs(estimated[a,0]-z[b,0]),abs(estimated[a,1]-z[b,1])*1e9,
                spherical_angle(directions(estimated[a,2:4]),directions(z[b,2:4]))*180/np.pi,
                spherical_angle(directions(estimated[a,4:6]),directions(z[b,4:6]))*180/np.pi])
        controlled.append({'case':i,'truth':z,'estimated':estimated,'quality':q,'squared_gospa':d,
                           'miss':miss,'false':false,'matched_errors_gain_db_delay_ns_directions_deg':error})
    queries,_=g.sample_safe_queries(32,93701);features=g.features(queries);labels=[];front=[];comparisons=[]
    for i,x in enumerate(queries):
        s,p=oracle.observe(x,(9001,901,i));z,q=rf.estimate(s,p);labels.append(z)
        reference=world.reference(x);d,pairs,miss,false=squared_gospa(z,reference,cfg)
        front.append({'index':i,'quality':q,'estimated_paths':z,'reference_paths':reference,'squared_gospa':d,'miss':miss,'false':false})
        traced=world.paths(x)[0][:,1];analytic=np.array([r[2]/cfg['propagation']['speed_of_light_mps'] for r in g.candidate_paths(x) if r[5]])
        discrepancy=float(np.min(abs(traced[:,None]-analytic[None,:]),axis=1).max()*1e9) if len(traced) and len(analytic) else None
        comparisons.append({'index':i,'sionna_count':len(traced),'analytic_count':len(analytic),'max_nearest_delay_error_ns':discrepancy})
    committee=Committee(cfg,9101);start=time.perf_counter();log=committee.train(features,labels,600)
    predictions=committee.predict(features);fit_error=np.mean([squared_gospa(a,b,cfg)[0] for a,b in zip(predictions,labels)])
    allfinite=all(np.isfinite(v['loss_end']) for v in log)
    matched_errors=[e for case in controlled[1:] for e in case['matched_errors_gain_db_delay_ns_directions_deg']]
    front_pass=all(case['quality']['order']==len(case['truth']) and case['miss']==case['false']==0 for case in controlled)
    if matched_errors:front_pass &= bool((np.median(matched_errors,axis=0)<[1.5,5,5,5]).all())
    fit_pass=all(v['loss_end']<v['loss_start']*.5 for v in log) and allfinite
    result={'scope':'development scene only','scientific_config_hash':scientific_hash(cfg),
        'material_model':world.material_document,'controlled_cases':controlled,'random_frontend':front,
        'ray_geometry_crosscheck':comparisons,'small_sample_fit':{'models':log,'wall_s':time.perf_counter()-start,
        'training_set_mean_squared_gospa':float(fit_error),'predicted_empty_fraction':float(np.mean([len(z)==0 for z in predictions]))},
        'frontend_passed':bool(front_pass),'small_sample_fit_passed':bool(fit_pass),'calibration':cal}
    write_json(out/'development_diagnostics.json',result)
    print('DEVELOPMENT frontend_passed=',front_pass,'small_sample_fit_passed=',fit_pass,'training_set_squared_gospa=',fit_error,flush=True)
    if not front_pass or not fit_pass:raise RuntimeError('Development diagnostics failed; inspect before freezing')
    return result
