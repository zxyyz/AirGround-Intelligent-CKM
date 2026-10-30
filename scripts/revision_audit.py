"""E0 read-only historical audit; all outputs stay in the revision directory."""
import sys,json,csv,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from paths import EXP,config,save,sha,stamp
from data import folds,blocks
from ckm_sim.geometry import GeometryMap
OLD=EXP.parent/'geometry_feature_attribution_20261027'
def run():
 p=np.load(OLD/'cache/pool.npz');x=p['x'];u,ix=np.unique(x,axis=0,return_inverse=True);ff=folds(u)
 assert np.array_equal(ff[ix],p['fold'])
 assert np.all(p['fold'][p['train']]<3)
 rows=[]
 for flow in sorted(np.unique(p['pair'])):
  for label,m in [('all_records',p['pair']==flow),('training',(p['pair']==flow)&p['train'])]:
   n=int(m.sum());un=len(np.unique(x[m],axis=0))
   rows.append(dict(dataset=label,flow=int(flow),records=n,unique_positions=un,empty_labels=int((p['n_paths'][m]==0).sum()),failures=int((~p['valid'][m]).sum()),blocks=len(np.unique(blocks(x[m]),axis=0)),repeat_observations=n-un))
 train=np.unique(x[p['train']],axis=0);g=GeometryMap(config());tree=cKDTree(train[:,:3]);sets={'training':set(map(tuple,train))};buffer_hits={}
 for name in ['dev','confirm']:
  q=np.load(OLD/f'cache/queries_{name}.npz')['queries'];ref=np.load(OLD/f'cache/reference_{name}.npz');expected=3 if name=='dev' else 4
  assert np.all(folds(q)==expected);sets[name]=set(map(tuple,q));hits=[]
  for i,point in enumerate(q):
   near=tree.query_ball_point(point[:3],10.)
   if near and np.any(g.road_distance(point[3:],train[near,3:])<=5.):hits.append(i)
  buffer_hits[name]=hits
  rows.append(dict(dataset=name,flow='shared',records=len(q),unique_positions=len(sets[name]),empty_labels=int((ref['n_paths']==0).sum()),failures=0,blocks=len(np.unique(blocks(q),axis=0)),repeat_observations=len(q)-len(sets[name])))
 intersections={a+'__'+b:len(sets[a]&sets[b]) for a,b in [('training','dev'),('training','confirm'),('dev','confirm')]}
 assert not any(intersections.values());assert not buffer_hits['confirm']
 out=dict(created_at=stamp(),passed=True,split_type='one fixed five-group spatial split, NOT five-fold cross-validation',block_widths_m=[60,60,30,60,60],training_folds=[0,1,2],development_fold=3,confirmation_fold=4,ugv_z_fixed=float(x[0,5]),same_position_fold_unique=True,intersection_counts=intersections,joint_buffer='UAV Euclidean <=10m AND road distance <=5m',joint_buffer_violations=buffer_hits,raw_records=len(x),raw_unique_positions=len(u),training_records=int(p['train'].sum()),training_unique_positions=len(train),map_shape=list(g.buildings.shape),map_columns=['xmin','ymin','xmax','ymax','height'],raw_map_tensor='6 pose coordinates + 64x40 tokens; 16 active building tokens, each 5 bounds + 12 relative corner coordinates + 23 zero padding',counts=rows,legacy_reference_failure_evidence='All archived reference chunks exist; original generator raises on invalid quality. No per-frame quality archive in consolidated NPZ; zero is inferred from successful generation, not a newly replayed extraction.')
 save(EXP/'results/split_audit.json',out)
 with (EXP/'results/data_counts.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 # SVG uses text and shapes, no plotting runtime required.
 svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="300" viewBox="0 0 1000 300"><rect width="1000" height="300" fill="white"/><g font-family="DejaVu Sans" fill="#163047"><text x="30" y="40" font-size="24">One fixed spatial split, shared across all noise flows</text><text x="30" y="75" font-size="17">Block widths: UAV (60, 60, 30 m); UGV (60, 60 m). UGV height fixed.</text>'''
 for i in range(5):
  col='#dbeafe' if i<3 else '#dcfce7' if i==3 else '#ffedd5'
  svg+=f'<rect x="{30+i*192}" y="105" width="178" height="90" rx="8" fill="{col}"/><text x="{45+i*192}" y="137" font-size="18">Hash group {i}</text><text x="{45+i*192}" y="171" font-size="17">'+('Training' if i<3 else 'Development' if i==3 else 'Confirmation')+'</text>'
 svg+='<text x="30" y="240" font-size="17">Confirmation exclusion: UAV distance ≤10 m AND road distance ≤5 m.</text><text x="30" y="272" font-size="17">Same position always shares its group across flows. This is not cross-validation.</text></g></svg>'
 (EXP/'figures/spatial_split.svg').write_text(svg)
 print(json.dumps({k:out[k] for k in ['passed','training_records','training_unique_positions','intersection_counts','joint_buffer_violations']}))
if __name__=='__main__':run()
