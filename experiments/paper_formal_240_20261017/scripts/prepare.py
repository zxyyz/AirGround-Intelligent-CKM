"""Independent evaluation assets and pre-result execution freeze."""
from common import *
import argparse,subprocess
import numpy as np,torch
from ckm_sim.geometry import GeometryMap
from formal_models import Features
from collector import Receiver
from physics import SignalWorld,fold
from ckm_sim.sounding import simulate_received
from ckm_sim.metrics import squared_gospa

def assets(phase):
    seed=2026101812 if phase=='smoke' else 2026101811;n=256 if phase=='smoke' else 4096
    cfg=config();g=GeometryMap(cfg);out=EXP/f'cache/query_{phase}.npz'
    if not out.exists():
        previous=[]
        old=ROOT/'experiments/geometry_conditioned_path_detection_20261017/cache'
        for name in ['queries_dev.npz','queries_confirm.npz','queries_main.npz']:previous.extend(np.load(old/name)['queries'])
        seen={tuple(a) for a in previous};q,w=g.sample_safe_queries(n,seed)
        if any(tuple(a) in seen for a in q):
            pool,_=g.sample_safe_queries(n*4,seed);q=np.array([a for a in pool if tuple(a) not in seen])[:n]
        assert len(q)==n
        np.savez_compressed(out,queries=q,weights=np.full(n,1/n))
        save(EXP/f'configs/query_freeze_{phase}.json',dict(seed=seed,n=n,sha256=sha(out),epoch=time.time(),reference_generated=False,selection='safe geometry and historical exact-coordinate exclusion only'))
    q=np.load(out)['queries'];featurefile=EXP/f'cache/features_{phase}.npz'
    if not featurefile.exists():
        fd=Features(cfg,'D');fg=Features(cfg,'M3');np.savez_compressed(featurefile,D=fd.features(q),M3=fg.features(q));save(EXP/f'results/query_geometry_{phase}.json',dict(candidate_counts=fg.costs))
    ref=EXP/f'cache/reference_{phase}.npz'
    if ref.exists():return
    torch.set_num_threads(2);world=SignalWorld('W1a',cfg);receiver=Receiver(cfg);signals=[];records=[];labels=[];masks=[];pcounts=[]
    for i,x in enumerate(q):
        h,rec,_,_=world.channel(x);s,p=simulate_received(h,(seed,0,i),cfg,noiseless=True);signals.append(s);records.append(rec)
        if len(signals)==8 or i==len(q)-1:
            zs,qs=receiver.estimate(np.stack(signals),p)
            for z,quality,r in zip(zs,qs,records):
                assert quality['valid'];labels.append(z)
                ix=[j for j in r['signal_indices'] if r['paths'][j]['z'][0]>=-110][:3]
                physical=fold(np.array([r['paths'][j]['z'] for j in ix]).reshape(-1,6));_,pairs,_,_=squared_gospa(z,physical,cfg);mask=np.zeros(3,bool)
                for oi,pi in pairs:mask[oi]=r['paths'][ix[pi]]['kind']=='1+1'
                masks.append(mask);pcounts.append([len(ix),sum(r['paths'][j]['kind']=='1+1' for j in ix)])
            signals=[];records=[]
        if (i+1)%256==0:print('REFERENCE',phase,i+1,flush=True)
    k=np.array([len(z) for z in labels]);z=np.zeros((n,3,6))
    for i,a in enumerate(labels):z[i,:len(a)]=a
    np.savez_compressed(ref,reference=z,n_paths=k,observable_second_mask=np.array(masks),physical_counts=np.array(pcounts),weights=np.full(n,1/n))
    save(EXP/f'results/reference_{phase}.json',dict(sha256=sha(ref),n=n,n_components=int(k.sum()),second_components=int(np.array(masks).sum()),world_manifest=world.audit.manifest,epoch=time.time()))

def freeze():
    assert json.loads((EXP/'runs/smoke/5099/complete.json').read_text())['status']=='complete'
    assert json.loads((EXP/'results/engineering_acceptance.json').read_text())['passed']
    for name in ['contract_tests.json','geometry_tests.json','candidate_level_tests.json','rng_namespace_audit.json']:
        assert json.loads((EXP/'results'/name).read_text())['passed']
    import importlib.metadata as metadata
    dependencies={}
    for name in ['torch','numpy','scipy','h5py','matplotlib','sionna-rt','mitsuba','drjit']:
        try:dependencies[name]=metadata.version(name)
        except metadata.PackageNotFoundError:dependencies[name]='not installed under this distribution name'
    save(EXP/'configs/dependencies.json',dict(python=sys.version,executable=sys.executable,packages=dependencies))
    protected=json.loads((EXP/'configs/protected_before.json').read_text())
    assert all(sha(ROOT/p)==v for p,v in protected.items())
    files=[*sorted((EXP/'scripts').glob('*.py')),EXP/'configs/effective_config.json',EXP/'configs/mesh_M2.json',EXP/'configs/calibration.json',EXP/'configs/protocol.plan.json',EXP/'configs/dependencies.json',EXP/'configs/environment/environment.json',*sorted((EXP/'assets').glob('*')),EXP/'cache/query_formal.npz',EXP/'cache/reference_formal.npz',EXP/'cache/features_formal.npz',*sorted((ROOT/'ckm_sim').glob('*.py'))]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
    for name in ['device_assignment.json','initialization_seed_map.json','preserved_M3_v1.json','preserved_measurements_v1.json']:
        p=EXP/'configs'/name;hashes[str(p.relative_to(ROOT))]=sha(p)
    previous=EXP/'archives/initialization_attempt_v1/freeze_v1.json'
    save(EXP/'configs/freeze.json',dict(version=2,previous_freeze_sha256=sha(previous),epoch=time.time(),seeds=list(range(5101,5111)),branches=['I-D','I-G','C-G','P-G'],hashes=hashes,net_sensing_s=14400,steps_per_member=7100,threshold=.5,raw_S_retention='first8 per window (stronger than first8 per stream), global_probe_index%100==0, all invalid frames',runtime_initial_slots=json.loads((EXP/'configs/runtime_slots.json').read_text()),runtime_allocation_changes_allowed=True,source='predeclared plan; initialization namespace correction before inspecting formal outcomes; D retrained, compatible M3/data retained'))
    print('FROZEN',len(hashes),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['smoke','formal','freeze']);a=p.parse_args()
    try:freeze() if a.phase=='freeze' else assets(a.phase)
    except Exception as e:failure('prepare',e);raise
