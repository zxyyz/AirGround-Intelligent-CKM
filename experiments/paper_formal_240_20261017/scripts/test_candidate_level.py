from common import *
import unittest,itertools,numpy as np,torch
from candidate_level import generalized_loss,CandidateLevelCKM
from ckm_sim.model import matched_set_loss
from ckm_sim.metrics import path_squared_distance
from ckm_sim.geometry import angles
from support import cfg_for
from decoding import decode
class Tests(unittest.TestCase):
 def fixture(self,n):
  torch.manual_seed(4);cfg=cfg_for('W1a');b=4;logit=torch.randn(b,n,requires_grad=True);gain=-100+torch.randn(b,n)*5;delay=torch.rand(b,n)*1e-6;v=[torch.nn.functional.normalize(torch.randn(b,n,3),dim=-1) for _ in range(2)];labels=torch.zeros(b,3,6);labels[...,0]=-98;labels[...,1]=3e-7;labels[...,3]=.6;labels[...,5]=2.3;lengths=torch.tensor([0,1,2,3]);return cfg,(logit,gain,delay,*v),labels,lengths
 def test_original_equivalence(self):
  cfg,out,z,k=self.fixture(3);a=generalized_loss(out,z,k,cfg);b=matched_set_loss(out,z,k,cfg);torch.testing.assert_close(a,b);a.backward();self.assertTrue(torch.isfinite(out[0].grad).all())
 def test_brute_injection(self):
  cfg,out,z,k=self.fixture(5);a=generalized_loss(out,z,k,cfg);l,g,t,rx,tx=[v.detach().numpy() for v in out];p=np.concatenate([g[...,None],t[...,None],angles(rx),angles(tx)],-1);vals=[]
  for j,n in enumerate(k.tolist()):
   d=path_squared_distance(p[j,:,None],z[j,:n].numpy()[None],cfg);best=min(sum(d[i,c]-l[j,i] for c,i in enumerate(pp)) for pp in itertools.permutations(range(5),n));vals.append(np.logaddexp(0,l[j]).sum()+best)
  self.assertAlmostEqual(float(a),float(np.mean(vals)),places=4)
 def test_decode_nms_cap(self):
  cfg=cfg_for('W1a');z=np.zeros((3,8,65,6));z[...,0]=-100;z[...,1]=1e-7;z[...,3]=.6;z[...,5]=2.3;l=np.ones((3,8,65));p=decode(z,l,.5,cfg);self.assertTrue(all(len(a)==1 for a in p));z[...,1]=np.arange(65)*20e-9;p=decode(z,l,.5,cfg);self.assertTrue(all(len(a)==3 for a in p))
if __name__=='__main__':
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));save(EXP/'results/candidate_level_tests.json',dict(tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),passed=r.wasSuccessful()));sys.exit(not r.wasSuccessful())
