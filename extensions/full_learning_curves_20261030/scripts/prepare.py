from common import *
import csv,subprocess
f=assert_freeze()
assert read(BASE/'configs/effective_config.json')['sounding']['delta_f_hz']==480000
assert read(BASE/'configs/protocol.json')['batch_size']==64
original=read(BASE/'manifests/formal_jobs.json');assert len(original)==140
prior_keys={(t['variant'],t['pair'],t['ratio']) for t in original}
rows=[]
for ratio in [.1,.25,.5]:
 for variant in ['GeoPool','GeoDirect']:
  for pair in range(5101,5111):
   assert (variant,pair,ratio) not in prior_keys
   lr=f['learning_rates'][variant];path=directory(variant,lr,pair,ratio)/'step_056800'
   basepath=BASE/f'runs/formal/ratio_{ratio:.2f}/{variant}/lr_{lr:g}/{pair}/step_056800'
   assert not basepath.exists(),('Unexpected existing extension artifact',basepath)
   rows.append(dict(variant=variant,pair=pair,ratio=ratio,lr=lr,threshold=f['thresholds'][variant],steps=56800,members=3,path=str(path.relative_to(EXT)),batch=str(BASE/f'manifests/batches/formal_{pair}_{ratio:.2f}.npz'),status='existing' if (path/'evaluation_confirm.json').exists() else 'pending'))
assert len(rows)==60
p=EXT/'manifests/tasks.json'
if p.exists():assert [{k:v for k,v in r.items() if k!='status'} for r in read(p)]==[{k:v for k,v in r.items() if k!='status'} for r in rows]
else:save(p,rows)
manifest=EXT/'audit/protected_original_files.json'
if not manifest.exists():
 files=[]
 for folder in ['scripts','support','ckm_sim','configs','results','manifests','runs','figures','paper']:
  files.extend(p for p in (BASE/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
 files.extend(p for p in (BASE/'cache').glob('*.npz'))
 files.extend(p for p in BASE.iterdir() if p.is_file())
 files.extend(p for p in (BASE/'release_candidate').iterdir() if p.is_file() and not p.name.endswith('.tar.gz'))
 save(manifest,dict(created_at=stamp(),sha256={str(p.relative_to(BASE)):sha(p) for p in sorted(set(files))},scope='Original scientific sources, all prior run artifacts, configurations, scientific NPZ caches, statistics, figures and manuscript. Compiler/runtime caches are isolated for this extension. Existing archives are addressed by unchanged SHA256SUMS.'))
else:
 for name,h in read(manifest)['sha256'].items():assert sha(BASE/name)==h,name
save(EXT/'audit/initial_inventory.json',dict(created_at=stamp(),existing_formal_groups=140,extension_tasks=60,extension_evaluations_found=sum(r['status']=='existing' for r in rows),baseline_freeze_sha256=sha(BASE/'configs/freeze.json'),batch_files_reused=30,no_hyperparameter_search=True))
save(EXT/'audit/extension_protocol.json',dict(created_at=stamp(),authorization='User explicitly authorized 60 optional groups on 2026-10-30 after reviewing initial confirmation results.',baseline=str(BASE),baseline_freeze_sha256=sha(BASE/'configs/freeze.json'),scope='GeoPool/GeoDirect x 10/25/50% x 5101-5110 x 3 members',formal_stage_namespace='formal (unchanged)',initializer='Original SetModel.initialize(formal,pair,member); each ratio initialized from scratch; no full-data checkpoint loading',output_adaptation='Only train.run_dir redirected; original science modules imported unchanged; pre-existing batch files mandatory',optimizer='Original AdamW lr .003/.0003; weight decay .0001; batch64; gradient clipping5; 56800 updates/member; float32 TF32 disabled',thresholds={'GeoPool':.9,'GeoDirect':.5},inference='Original threshold, NMS, max3 paths, three-member medoid and float64 GOSPA',validation='Every query batch replay; 80 cardinality-stratified scalar matches per new group (same as original) and every-query component identity; original source/results integrity checked again',analysis='New sparse contrasts are descriptive extensions after previous confirmation was viewed; no retuning; original four primary full-data tests unchanged',online_writes=False,services_changed=False))
print('PREPARED',len(rows),'new groups; protected',len(read(manifest)['sha256']),'original files',flush=True)
