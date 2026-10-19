"""Actual records, clone/sampling fairness, frozen code and final budget audit."""
from common import *
import numpy as np,torch,subprocess
from store import get_window,digest,locked

def run():
    frozen=json.loads((EXP/'configs/freeze.json').read_text());protected=json.loads((EXP/'configs/protected_before.json').read_text());checks={};total=0;rawtotal=0;seedrows=[]
    assert all(sha(ROOT/p)==v for p,v in frozen['hashes'].items());checks['frozen_code_assets_config_queries_unchanged']=True
    assert all(sha(ROOT/p)==v for p,v in protected.items());checks['protected_formal_source_paper_root_handoff_unchanged']=True
    correction=json.loads((EXP/'results/rng_correction.json').read_text())
    assert not correction['outcomes_inspected'] and correction['formal_evaluation_files_at_detection']==0
    assert json.loads((EXP/'results/rng_namespace_audit.json').read_text())['passed']
    preserved=json.loads((EXP/'configs/preserved_M3_v1.json').read_text())
    assert all(sha(ROOT/p)==v for p,v in preserved.items());checks['preserved_M3_checkpoints_bitwise_unchanged']=len(preserved)
    old_records=json.loads((EXP/'configs/preserved_measurements_v1.json').read_text())
    with locked('r') as h:
        assert all(h[p].attrs['digest']==v for p,v in old_records.items())
    checks['preserved_measurement_windows_unchanged']=len(old_records)
    cfg=config();assert cfg['scene']['ugv_antenna_height_m']==1.5 and cfg['propagation']['rt_max_depth']==2;assert cfg['propagation']['rx_zenith_interval_rad']==[0,np.pi] and cfg['propagation']['tx_zenith_interval_rad']==[0,np.pi]
    checks['effective_physics_config']=True
    for pair in frozen['seeds']:
        root=EXP/f'runs/formal/{pair}';assert (root/'complete.json').exists() and (root/'evaluation_complete.json').exists();initials=[];sampling_ok=True;local_raw=0
        for r in range(8):
            for stream in (['I'] if r==0 else ['I','C','P']):
                a,meta,value=get_window('formal',pair,stream,r,signals=True);n=len(a['x']);assert n==1800
                norm={k:np.array([v.decode() if isinstance(v,bytes) else v for v in a[k]]) if a[k].dtype.kind in 'OUS' else a[k] for k in a}
                assert digest(norm)==value
                assert np.all(a['x'][:,5]==1.5);np.testing.assert_array_equal(a['global_probe_index'],np.arange(r*1800,(r+1)*1800));np.testing.assert_allclose(a['time_s'],r*1800+np.arange(1800)+.5)
                want=np.flatnonzero((np.arange(1800)<8)|((r*1800+np.arange(1800))%100==0)|~a['valid']);np.testing.assert_array_equal(a['raw_index'],want);assert len(a['S'])==len(want)
                assert np.all((a['n_paths']>=0)&(a['n_paths']<=3));assert meta['service']['total_s']==427.96875 and meta['service']['sensing_probes']==0
                assert np.allclose(meta['state']['x'],np.r_[cfg['scene']['initial_uav_xyz_m'],cfg['scene']['initial_ugv_xyz_m']]);total+=n;rawtotal+=len(want);local_raw+=len(want)
            d=torch.load(root/'I-D'/f'checkpoints/update_{r:02d}.pt',map_location='cpu',weights_only=False);g=torch.load(root/'I-G'/f'checkpoints/update_{r:02d}.pt',map_location='cpu',weights_only=False)
            assert d['initialization_namespace']=='D-replay:init:SHA256-63bit:v2'
            assert d['record_digest']==g['record_digest'];assert [m['sample_index_sha256'] for m in d['training_log']]==[m['sample_index_sha256'] for m in g['training_log']]
            for branch in ['I-D','I-G','C-G','P-G']:
                s=d if branch=='I-D' else g if branch=='I-G' else torch.load(root/branch/f'checkpoints/update_{r:02d}.pt',map_location='cpu',weights_only=False)
                assert s['committee']['steps']==[1500+800*r]*3 and s['n_attempts']==1800*(r+1)
                met=json.loads((root/branch/f'metrics_{r:02d}.json').read_text());assert met['checkpoint_sha256']==sha(root/branch/f'checkpoints/update_{r:02d}.pt')
                assert len(np.load(root/branch/f'evaluation_{r:02d}.npz')['true_k'])==4096
                if r==0 and branch!='I-D':initials.append(s)
        assert len({s['weight_hash'] for s in initials})==1
        for a,b in zip(initials,initials[1:]):
            for x,y in zip(a['committee']['samplers'],b['committee']['samplers']):assert torch.equal(x,y)
            for x,y in zip(a['cuda_rng'],b['cuda_rng']):assert torch.equal(x,y)
            assert torch.equal(a['torch_rng'],b['torch_rng'])
        seedrows.append(dict(pair=pair,models=4,windows_per_model=8,attempts_per_model=14400,steps_per_member=7100,unique_attempts=39600,raw_signals=local_raw,initial_M3_hash=initials[0]['weight_hash']))
    assert total==396000
    service=subprocess.check_output(['systemctl','--user','show','llama-server.service','-p','ActiveState','-p','SubState','-p','MainPID'],text=True);assert 'ActiveState=inactive' in service and 'MainPID=0' in service
    checks.update(passed=True,pairs=seedrows,formal_model_results=40,formal_evaluation_points=320,unique_signal_attempts=total,retained_signal_frames=rawtotal,signal_retention_fraction=rawtotal/total,service_state=service,epoch=time.time(),scope='single fixed city; final frozen methods; all predefined seeds including negative results')
    save(EXP/'results/integrity.json',checks);print('FORMAL_INTEGRITY_PASS',total,rawtotal,flush=True)

if __name__=='__main__':
    try:run()
    except Exception as e:failure('audit',e);raise
