"""All 15 scales x320 fixed predictions; workers never modify legacy inputs."""
from common import *
from scoring import components,score
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse

def worker(pair):
    start=time.perf_counter();refd=np.load(OLD/'cache/reference_formal.npz');ref=refd['reference'];kr=refd['n_paths'];second=refd['observable_second_mask'];w=refd['weights'];grid=scales();rows=[];countrows=[]
    for branch in BRANCHES:
        for r in range(8):
            d=np.load(OLD/f'runs/formal/{pair}/{branch}/evaluation_{r:02d}.npz');pred=d['prediction'].astype(np.float32);kp=d['predicted_k'];assert np.array_equal(d['true_k'],kr)
            assert np.array_equal(pred.astype(np.float64),d['prediction']),'saved coordinates must exactly encode original float32 outputs'
            terms=components(pred,ref);base=np.load(EXP/f'cache/S0_assignment_{pair}_{branch}_{r:02d}.npz')['assignment_code'];conf=np.zeros((4,4),int);np.add.at(conf,(kr,kp),1)
            for a in range(4):
                for b in range(4):countrows.append(dict(pair=pair,branch=branch,round=r,K_ref=a,K_pred=b,count=int(conf[a,b])))
            masks=[];invariant=None
            for scale in grid:
                scored,detail=score(pred,kp,ref,kr,second,w,scale,terms=terms,baseline_codes=base)
                if scale['id']=='S0':np.testing.assert_array_equal(detail['assignment_code'],base)
                for v in scored:
                    row=dict(scale=scale['id'],pair=pair,branch=branch,round=r,minutes=30*(r+1),**v);rows.append(row)
                current=tuple((v['subset'],v.get('predicted_components'),v.get('reference_components'),v.get('count_accuracy')) for v in scored)
                if invariant is None:invariant=current
                assert current==invariant,'counts cannot change under fixed-prediction rescoring'
                masks.append(detail['assignment_code'])
            np.savez_compressed(EXP/f'cache/assignments_{pair}_{branch}_{r:02d}.npz',codes=np.array(masks),scales=np.array([s['id'] for s in grid]),K_pred=kp,K_ref=kr)
    csvwrite(EXP/f'results/scale_rows_{pair}.csv',rows);csvwrite(EXP/f'results/count_confusion_{pair}.csv',countrows)
    return dict(pair=pair,scales=len(grid),updates=32,scored_set_pairs=32*4096*len(grid),wall_s=time.perf_counter()-start,all_count_invariants=True)

def run(workers):
    assert load(EXP/'results/baseline_reproduction.json')['passed'];assert load(EXP/'results/physical_tests.json')['passed'];assert load(EXP/'results/scoring_tests.json')['passed']
    protocol=load(EXP/'configs/protocol_freeze.json');assert sha(EXP/'configs/protocol.json')==protocol['sha256']
    stage('C_full_scale_rescoring');start=time.perf_counter();evidence=[]
    # Freeze implementation and resolved physical scales before new-scale results.
    freeze={str(p.relative_to(EXP)):sha(p) for p in sorted((EXP/'scripts').glob('*.py'))};freeze['configs/scales_resolved.json']=sha(EXP/'configs/scales_resolved.json');save(EXP/'configs/scoring_source_freeze.json',dict(epoch=time.time(),hashes=freeze,scale_outcomes_inspected=False))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(worker,p):p for p in range(5101,5111)}
        for future in as_completed(jobs):
            evidence.append(future.result());save(EXP/'results/rescoring_progress.json',dict(completed=evidence,wall_s=time.perf_counter()-start));print('RESCORED_PAIR',evidence[-1]['pair'],len(evidence),'/10',flush=True)
    save(EXP/'results/rescoring_complete.json',dict(passed=True,workers=workers,pairs=evidence,scales=15,model_updates=320,scale_updates=4800,scored_set_pairs=19660800,wall_s=time.perf_counter()-start,epoch=time.time(),counts_invariant=True));stage('C_scoring_complete_statistics_pending',C='scored')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=10);a=p.parse_args()
    try:run(a.workers)
    except Exception as e:failure('rescore',e);raise
