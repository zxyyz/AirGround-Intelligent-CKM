from common import *
import unittest,numpy as np
from support import cfg_for
from public_multibounce_geometry import PublicGeometry,FIELDS,D,canonical_vectors
from ckm_sim.geometry import GeometryMap
class Tests(unittest.TestCase):
 def setUp(self):self.cfg=cfg_for('W1a');self.g=PublicGeometry(self.cfg)
 def test_g0_exact(self):
  x,_=self.g.g.sample_safe_queries(32,791);np.testing.assert_array_equal(np.stack([self.g.g0(a) for a in x]),self.g.g.features(x))
 def test_clear_vectorized(self):
  rng=np.random.default_rng(2);a=rng.uniform([0,0,0],[240,240,120],(256,3));b=rng.uniform([0,0,0],[240,240,120],(256,3));np.testing.assert_array_equal(self.g.clear_many(a,b),[self.g.g.clear(c,d) for c,d in zip(a,b)])
 def test_order_and_laws(self):
  x,_=self.g.g.sample_safe_queries(64,792);orders=set();double=0
  for a in x:
   cc,st=self.g.candidates(a,2);cc2,_=self.g.candidates(a,2);self.assertEqual([(r['ids'],r['length']) for r in cc],[(r['ids'],r['length']) for r in cc2]);self.assertEqual(len(FIELDS),D)
   for r in cc:
    if not r['valid']:continue
    orders.add(r['order']);p=r['vertices'];seg=np.diff(p,axis=0);seg/=np.linalg.norm(seg,axis=-1,keepdims=True)
    self.assertTrue(self.g.clear_many(p[:-1],p[1:]).all());self.assertTrue(np.isfinite(r['row']).all());self.assertGreaterEqual(r['row'][5],0);self.assertLessEqual(r['row'][8],0)
    for j,f in enumerate(r['ids']):
     self.assertTrue(self.g.finite(p[j+1:j+2],np.array([f]))[0]);n=self.g.normals[f];np.testing.assert_allclose(seg[j+1],seg[j]-2*(seg[j]@n)*n,atol=1e-6)
    double+=r['order']==2
  self.assertEqual(orders,{0,1,2});self.assertGreater(double,0)
 def test_known_two_parallel_walls(self):
  # x=0 (normal+x), x=10 (normal-x); unfolded path length sqrt(25^2+2^2).
  gg=GeometryMap(self.cfg);gg.buildings=np.array([[-1,-100,0,100,100],[10,-100,11,100,100]],float);gg.faces=[(0,0,0,0,1),(1,0,0,10,-1)];gg.faces=[(0,0,0,0,1),(1,0,0,10,-1)];pg=PublicGeometry(self.cfg,gg);x=np.array([2,0,20,7,2,20.]);f1,f2,p1,p2=pg.double_vertices(x);ids=np.flatnonzero((f1==0)&(f2==1));self.assertEqual(len(ids),1);j=ids[0];r=pg.make(x,2,[0,1],[p1[j],p2[j]]);self.assertTrue(r['valid']);self.assertAlmostEqual(r['length'],np.sqrt(15**2+2**2))
 def test_finite_and_boundaries(self):
  f=0;b=self.g.g.buildings[0];p=np.array([[b[0],(b[1]+b[3])/2,b[4]],[b[0],b[3]+1,1],[b[0],b[1],-1.] ]);np.testing.assert_array_equal(self.g.finite(p,np.array([f]*3)),[True,False,False]);self.assertTrue(self.g.clear_many([[0,0,1]],[[0,0,1]])[0])
 def test_blocker_reject(self):
  self.assertFalse(self.g.clear_many([[0,30,1]],[[60,30,1]])[0])
if __name__=='__main__':
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));save(EXP/'results/geometry_tests.json',dict(tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),passed=r.wasSuccessful()));sys.exit(not r.wasSuccessful())
