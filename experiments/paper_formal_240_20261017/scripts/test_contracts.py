from common import *
import unittest,ast,tempfile
import numpy as np,torch
from formal_models import make_model,Features
from runner import checkpoint,restore
from collector import observe,worker_init,Receiver
from store import get_window,history
from ckm_sim.geometry import GeometryMap
from ckm_sim.model import SetCKM
from ckm_sim.experiment import service_record,weight_hash
from ckm_sim.v14_planning import initial_state

class Contracts(unittest.TestCase):
    def test_effective_physics_budget_service(self):
        c=config();g=GeometryMap(c)
        self.assertEqual(c['propagation']['rt_max_depth'],2);self.assertEqual(c['scene']['ugv_antenna_height_m'],1.5)
        self.assertEqual(c['propagation']['rx_zenith_interval_rad'],[0,np.pi]);self.assertEqual(c['propagation']['tx_zenith_interval_rad'],[0,np.pi])
        self.assertEqual(c['experiment']['rounds_after_initial'],7)
        self.assertEqual(c['training']['initialization_steps']+7*c['training']['update_steps_per_round'],7100)
        with tempfile.TemporaryDirectory(dir=EXP/'cache') as d:self.assertEqual(service_record(Path(d),0,initial_state(c),c,g),427.96875)

    def test_architecture_sampling_and_deep_clone(self):
        c=config();torch.set_num_threads(2);g=GeometryMap(c);q,_=g.sample_safe_queries(12,5099)
        z=[np.array([[-90,1e-7,0,.6,0,2.5]]*int(i%3),dtype=float).reshape(-1,6) for i in range(12)]
        a=make_model(c,5099,'D');b=make_model(c,5099,'M3')
        self.assertTrue(all(type(m) is SetCKM for m in a.models));a.poses=q;b.poses=q
        al=a.train(Features(c,'D').features(q),z,1);bl=b.train(Features(c,'M3').features(q),z,1)
        self.assertEqual([m['sample_index_sha256'] for m in al],[m['sample_index_sha256'] for m in bl])
        with tempfile.TemporaryDirectory(dir=EXP/'cache') as d:
            saved=checkpoint(Path(d)/'test.pt',b,initial_state(c),0,'I',{},bl,'synthetic')
            other=make_model(c,5099,'M3');restore(other,saved);self.assertEqual(weight_hash(b),weight_hash(other))
            before=b.models[0].encoder[0].weight.detach().clone()
            with torch.no_grad():other.models[0].encoder[0].weight.add_(1)
            torch.testing.assert_close(b.models[0].encoder[0].weight,before);self.assertEqual(saved['committee']['steps'],[1,1,1])

    def test_worker_evaluation_access(self):
        for name in ['runner.py','collector.py','formal_models.py','decoding.py','models.py','candidate_level.py','coverage.py','public_multibounce_geometry.py']:
            tree=ast.parse((EXP/'scripts'/name).read_text())
            imports=[]
            for n in ast.walk(tree):
                if isinstance(n,ast.ImportFrom):imports.append(n.module or '')
                elif isinstance(n,ast.Import):imports.extend(a.name for a in n.names)
            self.assertFalse(any(v in ['evaluate','prepare','scoring','diagnose'] for v in imports),(name,imports))
            if name in ['formal_models.py','decoding.py','models.py','candidate_level.py','public_multibounce_geometry.py']:
                self.assertNotIn('SignalWorld',(EXP/'scripts'/name).read_text())

def acceptance():
    c=config();pair=5099;phase='smoke';root=EXP/f'runs/{phase}/{pair}';assert (root/'complete.json').exists()
    checks={};initial=[]
    for b in ['I-G','C-G','P-G']:
        p=root/b/'checkpoints/update_00.pt';s=torch.load(p,map_location='cpu',weights_only=False);initial.append(s)
        assert s['committee']['steps']==[1500]*3
    assert len({s['weight_hash'] for s in initial})==1
    for b in ['I-D','I-G','C-G','P-G']:
        s=torch.load(root/b/'checkpoints/update_01.pt',map_location='cpu',weights_only=False);assert s['committee']['steps']==[2300]*3 and s['n_attempts']==3600
    for r in range(2):
        a=torch.load(root/'I-D'/f'checkpoints/update_{r:02d}.pt',map_location='cpu',weights_only=False);b=torch.load(root/'I-G'/f'checkpoints/update_{r:02d}.pt',map_location='cpu',weights_only=False)
        assert [m['sample_index_sha256'] for m in a['training_log']]==[m['sample_index_sha256'] for m in b['training_log']]
    checks['common_M3_and_I_sampling']=True
    # Deterministic RF regeneration on the same hardware: inspect fixed indices,
    # including a noninitial point in every stream, never selected by performance.
    worker_init(c);receiver=Receiver(c);errors=[]
    for stream,r in [('I',0),('I',1),('C',1),('P',1)]:
        data,meta,d=get_window(phase,pair,stream,r,signals=True)
        ix=np.array([0,100,900]);signals,counts,_=observe((data['x'][ix],pair,r,ix))
        np.testing.assert_array_equal(data['raw_index'],np.arange(1800))
        old=data['S'][ix];err=float(np.max(abs(signals-old)));np.testing.assert_allclose(signals,old,rtol=1e-6,atol=1e-16)
        cached_signals,_,_=observe((data['x'][ix],pair,r,ix));np.testing.assert_array_equal(signals,cached_signals)
        zs,qs=receiver.estimate(signals,np.eye(8));np.testing.assert_array_equal([len(z) for z in zs],data['n_paths'][ix])
        for z,i in zip(zs,ix):np.testing.assert_allclose(z,data['labels'][i,:len(z)],rtol=1e-6,atol=1e-9)
        assert abs(meta['service']['total_s']-427.96875)<1e-9
        errors.append(dict(stream=stream,round=r,max_signal_abs_error=err,labels_match=True,lossless_channel_cache_repeat_exact=True))
    checks['signal_and_label_regeneration']=errors
    checks['eight_window_ledger']=dict(attempts=8*1800,steps=1500+7*800,services=8,physical_s=8*(1800+427.96875))
    checks['passed']=True;checks['epoch']=time.time();save(EXP/'results/engineering_acceptance.json',checks);print('ENGINEERING_ACCEPTED',flush=True)

if __name__=='__main__':
    if '--acceptance' in sys.argv:acceptance()
    else:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
        save(EXP/'results/contract_tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),epoch=time.time()))
        if not result.wasSuccessful():sys.exit(1)
