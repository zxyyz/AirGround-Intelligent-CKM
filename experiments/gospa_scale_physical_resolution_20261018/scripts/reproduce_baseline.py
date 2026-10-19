"""Recompute S0 for all 320 fixed medoid sets before new-scale conclusions."""
from common import *
from scoring import score

def run():
    stage('A_baseline_reproduction');start=time.perf_counter();reference=np.load(OLD/'cache/reference_formal.npz');ref=reference['reference'];kr=reference['n_paths'];second=reference['observable_second_mask'];w=reference['weights'];s0=scales()[0]
    assert np.array_equal(w,np.full(4096,1/4096));rows=[];old_rounds={(int(d['pair']),d['branch'],int(d['round'])):d for d in csv.DictReader((OLD/'results/round_metrics.csv').open())};values={}
    compare=['global_gospa','gospa_p90','matched_parameter_cost','miss_cost','false_cost','matched_recall','precision','mean_miss','mean_false','n_matched','reference_components','predicted_components','second_component_total','second_component_matched','second_matched_recall','count_accuracy',*['rmse_'+a for a in ['gain_db','delay_ns','aoa_az_deg','aoa_zen_deg','aod_az_deg','aod_zen_deg']]]
    for pair in range(5101,5111):
        for branch in BRANCHES:
            for r in range(8):
                base=OLD/f'runs/formal/{pair}/{branch}';d=np.load(base/f'evaluation_{r:02d}.npz');assert np.array_equal(d['true_k'],kr)
                scored,detail=score(d['prediction'],d['predicted_k'],ref,kr,second,w,s0);old={a['subset']:a for a in load(base/f'metrics_{r:02d}.json')['metrics']};maximum=0.
                for row in scored:
                    for key in compare:
                        if key not in old[row['subset']]:continue
                        a,b=row[key],old[row['subset']][key]
                        if np.isnan(a) and np.isnan(b):continue
                        err=abs(a-b);assert err<1e-8,(pair,branch,r,row['subset'],key,a,b,err);maximum=max(maximum,err)
                    if row['subset']=='all':
                        for key in compare:
                            if key in old_rounds[pair,branch,r]:
                                a,b=row[key],float(old_rounds[pair,branch,r][key]);assert (np.isnan(a) and np.isnan(b)) or abs(a-b)<1e-8
                        values[pair,branch,r]=row['global_gospa']
                error=float(np.max(abs(detail['squared_gospa']-d['decomposition'][:,0])));assert error<1e-8;maximum=max(maximum,error)
                np.savez_compressed(EXP/f'cache/S0_assignment_{pair}_{branch}_{r:02d}.npz',assignment_code=detail['assignment_code'])
                rows.append(dict(pair=pair,branch=branch,round=r,max_abs_error=maximum,passed=True,old_global=old['all']['global_gospa'],reproduced_global=values[pair,branch,r],prediction_sha256=sha(base/f'evaluation_{r:02d}.npz')))
        print('BASELINE_REPRODUCED',pair,flush=True)
    summary={b:float(np.mean([values[p,b,7] for p in range(5101,5111)])) for b in BRANCHES}
    effects={f'{a}/{b}':float(np.mean([100*(1-values[p,a,7]/values[p,b,7]) for p in range(5101,5111)])) for a,b in [('I-G','I-D'),('C-G','I-G')]}
    original=load(OLD/'results/paired_summary.json')['comparisons'];assert abs(effects['I-G/I-D']-original['geometry']['endpoint_relative']['mean'])<1e-8;assert abs(effects['C-G/I-G']-original['cooperation']['endpoint_relative']['mean'])<1e-8
    csvwrite(EXP/'results/baseline_reproduction.csv',rows);save(EXP/'results/baseline_reproduction.json',dict(passed=True,updates=len(rows),max_abs_error=max(r['max_abs_error'] for r in rows),endpoint_means=summary,effects=effects,wall_s=time.perf_counter()-start,epoch=time.time(),all_subsets_and_original_round_csv=True));stage('A_complete_B_pending',A='passed');print('BASELINE_ACCEPTED',summary,effects,flush=True)

if __name__=='__main__':
    try:run()
    except Exception as e:failure('baseline',e);raise
