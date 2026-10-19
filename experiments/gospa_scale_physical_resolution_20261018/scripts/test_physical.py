from common import *
import unittest
from derive_physical_scales import calculate,correlation
from ckm_sim.sounding import array_response

class Physical(unittest.TestCase):
    def test_actual_uca_formula_and_normalization(self):
        c=cfg();rng=np.random.default_rng(2026101802);a=rng.uniform([[-np.pi,0]],[[np.pi,np.pi]],(31,2));phi=2*np.pi*np.arange(8)/8;rr=1/(4*np.sin(np.pi/8))
        for receive in [True,False]:
            expected=np.exp((1j if receive else -1j)*2*np.pi*rr*np.sin(a[:,1,None])*np.cos(a[:,0,None]-phi))/np.sqrt(8)
            np.testing.assert_allclose(array_response(a,8,c,receive),expected,atol=3e-15);np.testing.assert_allclose(np.sum(abs(expected)**2,-1),1,atol=1e-14)
    def test_half_power_roots_and_units(self):
        d=calculate();c=cfg();h=np.deg2rad(d['azimuth_half_power_half_width_deg']);zen=np.deg2rad(d['zenith_toward_pole_half_width_deg'])
        self.assertAlmostEqual(float(correlation(h,np.pi/2,0,np.pi/2,c)),.5,places=12)
        self.assertAlmostEqual(float(correlation(0,np.pi/2-zen,0,np.pi/2,c)),.5,places=12)
        self.assertAlmostEqual(d['azimuth_HPBW_full_width_deg'],2*d['azimuth_half_power_half_width_deg'],places=12)
        self.assertAlmostEqual(d['nominal_delay_scale_ns'],15.625,places=12);self.assertAlmostEqual(d['Q_delay_scale_ns'],31.25,places=12)
        self.assertEqual(d['endpoint_tone_span_hz'],63.5e6)
        self.assertLess(abs(d['azimuth_HPBW_full_width_deg']-31.544),.01);self.assertLess(abs(d['zenith_toward_pole_half_width_deg']-43.5),.1)
        self.assertIsNone(d['direction_dependence'][0]['azimuth_full_width_deg'])
        widths=[r['azimuth_full_width_deg'] for r in d['direction_dependence'] if r['reference_zenith_deg']>=30];self.assertTrue(all(a>b for a,b in zip(widths,widths[1:])))
    def test_mirror_and_tx_rx_correlation(self):
        c=cfg();phi=np.linspace(0,np.pi,71);theta=.7
        np.testing.assert_allclose(correlation(phi,theta,0,theta,c),correlation(phi,theta,0,theta,c,False),atol=1e-14)
        a=np.c_[phi,np.full(len(phi),theta)];b=a.copy();b[:,1]=np.pi-theta
        np.testing.assert_allclose(array_response(a,8,c),array_response(b,8,c),atol=3e-15)

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Physical));save(EXP/'results/physical_tests.json',dict(passed=r.wasSuccessful(),tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),epoch=time.time()));sys.exit(0 if r.wasSuccessful() else 1)
