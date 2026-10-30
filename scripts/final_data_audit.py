import numpy as np
from scipy.spatial import cKDTree
from paths import EXP,config,save,stamp
from data import folds,blocks,subset_indices
from ckm_sim.geometry import GeometryMap
from revision_metrics import csvout
p=np.load(EXP/'cache/pool.npz');q={n:np.load(EXP/f'cache/queries_{n}.npz')['queries'] for n in ['dev','confirm']};train=np.unique(p['x'][p['train']],axis=0);g=GeometryMap(config());tree=cKDTree(train[:,:3]);rows=[]
assert np.all(folds(p['x'][p['train']])<3)
for flow in range(5101,5111):
 ix=np.flatnonzero(p['train']&(p['pair']==flow));u=np.unique(p['x'][ix],axis=0);rows.append(dict(dataset='training',flow=flow,records=len(ix),unique_positions=len(u),empty_labels=int((p['n_paths'][ix]==0).sum()),failed=int((~p['valid'][ix]).sum()),blocks=len(np.unique(blocks(u),axis=0)),repeat_observations=len(ix)-len(u)))
 previous=set()
 for ratio in [.1,.25,.5,1]:
  ids=set(subset_indices(p,flow,ratio));assert previous<=ids;previous=ids
for name,x in q.items():
 assert (folds(x)==(3 if name=='dev' else 4)).all();assert not set(map(tuple,x))&set(map(tuple,train))
 for point,near in zip(x,tree.query_ball_point(x[:,:3],10)):
  assert not near or not np.any(g.road_distance(point[3:],train[near,3:])<=5)
 r=np.load(EXP/f'cache/reference_{name}.npz');rows.append(dict(dataset=name,flow='shared',records=len(x),unique_positions=len(np.unique(x,axis=0)),empty_labels=int((r['n_paths']==0).sum()),failed=0,blocks=len(np.unique(blocks(x),axis=0)),repeat_observations=0))
assert not set(map(tuple,q['dev']))&set(map(tuple,q['confirm']))
csvout(EXP/'results/final_data_counts.csv',rows);save(EXP/'results/final_split_audit.json',dict(passed=True,counts=rows,same_position_across_flow_grouping_verified=True,intersections=0,joint_near_exclusions=0,nested_subsets_verified=True,created_at=stamp()))
print('Final split, shared record counts, and nested subsets verified.')
