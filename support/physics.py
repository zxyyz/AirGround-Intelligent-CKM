from common import *
import numpy as np
from rt_world import AuditWorld,filtered
from ckm_sim.sounding import channel_from_paths,simulate_received
from ckm_sim.music import RFEstimator

def fold(z):
    z=np.array(z,dtype=float).reshape(-1,6).copy()
    z[:,3]=np.minimum(z[:,3],np.pi-z[:,3]);z[:,5]=np.maximum(z[:,5],np.pi-z[:,5])
    return z

def frontend(cfg):
    p=EXP/'configs/calibration.json'
    cal=json.loads(p.read_text());assert abs(cal['normalized_rank_threshold']-cfg['estimator']['rank_threshold_value'])<1e-10
    return RFEstimator(cfg,cal),cal

class SignalWorld:
    def __init__(self,label,cfg=None):
        self.label=label;self.cfg=cfg or config()
        m=None if label=='W0' else json.loads((EXP/('configs/mesh_W1.json' if label=='W1' else 'configs/mesh_M2.json')).read_text())
        self.audit=AuditWorld('H0',label,m);self.geometry=self.audit.geometry
        self.mode='R0' if label=='W0' else 'R1'
    def channel(self,x):
        r=self.audit.trace(x,self.mode)
        ix=filtered(r,cfg=self.cfg,threshold=-140,fov=self.label=='W0',cap=6)
        z=np.array([r['paths'][i]['z'] for i in ix]).reshape(-1,6)
        a=np.array([complex(*r['paths'][i]['coefficient']) for i in ix],dtype=complex)*np.exp(-2j*np.pi*self.cfg['propagation']['carrier_hz']*z[:,1])
        h=channel_from_paths(z,a,self.cfg)
        r.update(signal_indices=ix,signal_physical_k=len(ix),target_physical_k=sum(p[0]>=-110 for p in z),ground_in_signal=sum(any(t['material'] in ['audit_ground','audit_asphalt'] for t in r['paths'][i]['hits']) for i in ix))
        return h,r,z,a

def fit_coefficients(h,z,cfg):
    from ckm_sim.sounding import joint_atom
    if not len(z):return np.empty(0,complex),float(np.linalg.norm(h)>0)
    atoms=np.stack([joint_atom(p[1:],cfg,cfg['sounding']['n_frequencies']) for p in z],axis=1)
    target=h.swapaxes(-1,-2).reshape(-1);a=np.linalg.lstsq(atoms,target,rcond=cfg['estimator']['numerical_rcond'])[0]
    return a,float(np.linalg.norm(atoms@a-target)**2/max(np.linalg.norm(target)**2,1e-60))
