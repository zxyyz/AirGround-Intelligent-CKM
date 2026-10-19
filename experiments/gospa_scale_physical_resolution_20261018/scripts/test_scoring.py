from common import *
import unittest
from scoring import components,match,score
from ckm_sim.metrics import squared_gospa,path_squared_distance

class TestScoring(unittest.TestCase):
    def test_all_cardinalities_random_against_original(self):
        rng=np.random.default_rng(2026101801);c=cfg();scale=scales()[0]
        for cutoff in [2.,3.,4.]:
            scale=copy.deepcopy(scale);scale['cutoff']=cutoff;c['metric']['gospa_c']=cutoff
            for kr in range(4):
                for kp in range(4):
                    a=np.zeros((31,3,6));b=np.zeros_like(a)
                    for z in [a,b]:
                        z[...,0]=rng.uniform(-110,-70,(31,3));z[...,1]=rng.uniform(0,1e-6,(31,3));z[...,2:]=rng.uniform(-np.pi,np.pi,(31,3,4));z[...,3]=rng.uniform(0,np.pi/2,(31,3));z[...,5]=rng.uniform(np.pi/2,np.pi,(31,3))
                    if kp and kr:a[:15,0]=b[:15,0]+np.array([.5,1e-9,1e-3,1e-3,1e-3,1e-3])
                    term=components(a,b);out=match(term,np.full(31,kp),np.full(31,kr),scale)
                    np.testing.assert_allclose(term[...,0]/25+term[...,1]/(25e-9)**2+term[...,2:].sum(-1)/np.deg2rad(15)**2,path_squared_distance(a[:,:,None],b[:,None,:],c),rtol=1e-13,atol=1e-10)
                    for q in range(31):
                        val,pairs,miss,false=squared_gospa(a[q,:kp],b[q,:kr],c)
                        self.assertAlmostEqual(val,out['squared_gospa'][q],places=10)
                        self.assertEqual(sum(1<<(i*3+j) for i,j in pairs),out['assignment_code'][q]);self.assertEqual(miss,out['nmiss'][q]);self.assertEqual(false,out['nfalse'][q])
    def test_tie_empty_wrapping_poles(self):
        s=copy.deepcopy(scales()[0]);a=np.zeros((1,3,6));b=a.copy();a[0,0,0]=15
        out=match(components(a,b),np.array([1]),np.array([1]),s);self.assertEqual(out['nmatch'][0],0);self.assertEqual(out['squared_gospa'][0],9)
        a[0,0]=[-90,1e-7,np.pi-1e-8,.5,-np.pi+1e-8,2.5];b[0,0]=[-90,1e-7,-np.pi+1e-8,.5,np.pi-1e-8,2.5]
        rows,d=score(a,np.array([1]),b,np.array([1]),np.zeros((1,3),bool),np.ones(1),s);self.assertLess(rows[0]['global_gospa'],1e-6)
        a[0,0,3]=b[0,0,3]=0;a[0,0,5]=b[0,0,5]=np.pi
        rows,d=score(a,np.array([1]),b,np.array([1]),np.zeros((1,3),bool),np.ones(1),s);self.assertEqual(rows[0]['n_aoa_az_deg'],0);self.assertEqual(rows[0]['n_aod_az_deg'],0)
    def test_uca_mirror_representatives(self):
        from ckm_sim.sounding import array_response
        c=cfg();a=np.array([.6,.4]);b=np.array([.6,np.pi-.4])
        np.testing.assert_allclose(array_response(a,8,c),array_response(b,8,c),atol=1e-14)
        # Scores operate on already-folded saved representatives, not physical mirror angles.
        self.assertAlmostEqual(min(a[1],np.pi-a[1]),min(b[1],np.pi-b[1]),places=14)

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestScoring));save(EXP/'results/scoring_tests.json',dict(passed=r.wasSuccessful(),tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),epoch=time.time()));sys.exit(0 if r.wasSuccessful() else 1)
