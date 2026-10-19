from common import *
from scoring import assignments,score
from analyze import interval
from ckm_sim.metrics import path_squared_distance

def medoid(paths,k,scale):
    sums=np.zeros((3,len(k[0])))
    c=cfg();c['metric'].update(gain_scale_db=scale['gain_db'],delay_scale_s=scale['delay_ns']*1e-9,aoa_direction_scale_deg=scale['angle_deg'],aod_direction_scale_deg=scale['angle_deg'],gospa_c=scale['cutoff'])
    for a,b in [(0,1),(0,2),(1,2)]:
        # D1 changes decoding, so preserve the legacy member/member arithmetic,
        # including float32 direction responses on both sides of the distance.
        cost=path_squared_distance(paths[a,:, :,None],paths[b,:,None,:],c);cut=scale['cutoff']**2;dist=cut*(k[a]+k[b])/2.
        for ka in range(4):
            for kb in range(4):
                ix=np.flatnonzero((k[a]==ka)&(k[b]==kb));best=dist[ix].copy()
                for pairs in assignments(ka,kb):
                    if not pairs:continue
                    value=sum(cost[ix,i,j] for i,j in pairs)+cut*(ka+kb-2*len(pairs))/2
                    keep=value<best-1e-12;best[keep]=value[keep]
                dist[ix]=best
        sums[a]+=dist;sums[b]+=dist
    return sums.argmin(0)

def run():
    assert load(EXP/'results/member_regeneration.json')['passed'];stage('D1_medoid_reselection');start=time.perf_counter();grid={s['id']:s for s in scales()};ref=np.load(OLD/'cache/reference_formal.npz');rows=[];changes=[]
    for pair in range(5101,5111):
        for branch in BRANCHES:
            data=np.load(EXP/f'cache/full_members_{pair}_{branch}.npz');paths=data['paths'];k=data['n_paths'];q=np.arange(4096);old=data['old_medoid'];old_selected=paths[old,q];oldk=k[old,q]
            selections={name:medoid(paths,k,grid[name]) for name in ['S0','S_phys']}
            assert np.array_equal(selections['S0'],old),(pair,branch,'S0 medoid does not reproduce')
            np.testing.assert_array_equal(old_selected,data['old_prediction'])
            np.testing.assert_array_equal(oldk,data['old_predicted_k'])
            for selection,ids in selections.items():
                pred=paths[ids,q];kp=k[ids,q];changed=ids!=old;setchanged=np.any(pred!=old_selected,axis=(1,2))|(kp!=oldk)
                changes.append(dict(pair=pair,branch=branch,selection_scale=selection,medoid_index_change_fraction=float(changed.mean()),prediction_set_change_fraction=float(setchanged.mean()),predicted_count_change_fraction=float((kp!=oldk).mean())))
                for scoring in ['S0','S_phys']:
                    scored,detail=score(pred,kp,ref['reference'],ref['n_paths'],ref['observable_second_mask'],ref['weights'],grid[scoring])
                    rows.extend(dict(pair=pair,branch=branch,selection_scale=selection,scoring_scale=scoring,**r) for r in scored)
        print('MEDOID_RESELECTED',pair,flush=True)
    allrows=[r for r in rows if r['subset']=='all'];lookup={(r['pair'],r['branch'],r['selection_scale'],r['scoring_scale']):r for r in allrows};summaries=[]
    for selection in ['S0','S_phys']:
        for scoring in ['S0','S_phys']:
            for comparison,a,b in [('geometry','I-G','I-D'),('cooperation','C-G','I-G'),('disagreement','P-G','C-G')]:
                values=[100*(1-lookup[pair,a,selection,scoring]['global_gospa']/lookup[pair,b,selection,scoring]['global_gospa']) for pair in range(5101,5111)]
                summaries.append(dict(selection_scale=selection,scoring_scale=scoring,comparison=comparison,**interval(values)))
    csvwrite(EXP/'results/medoid_reselection_changes.csv',changes);csvwrite(EXP/'results/medoid_cross_scores.csv',rows);csvwrite(EXP/'results/medoid_paired_summary.csv',summaries)
    save(EXP/'results/medoid_reselection.json',dict(passed=True,epoch=time.time(),scope='10 seeds x4 branches, endpoint only; regenerated full three-member sets; no retraining',S0_reproduction=True,wall_s=time.perf_counter()-start,comparisons=summaries));stage('D1_complete_report_pending',D1='complete');print('D1_COMPLETE',flush=True)

if __name__=='__main__':
    try:run()
    except Exception as e:failure('D1_medoid',e);raise
