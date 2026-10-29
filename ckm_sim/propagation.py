"""Private NVIDIA Sionna RT world. Controller never receives Paths or materials."""
import time
from pathlib import Path
import numpy as np
from .config import ROOT, array_hash
from .geometry import GeometryMap
from .sounding import channel_from_paths, simulate_received

def extract_reference_set(paths,cfg):
    t=cfg['ckm_target']
    keep=(paths[:,0]>=t['gain_min_db']) & (paths[:,1]<=t['max_delay_s'])
    return paths[keep][:t['max_paths']].copy()

def write_box_ply(path,b):
    lo=[b[0],b[1],0];hi=[b[2],b[3],b[4]]
    v=np.array([[lo[0],lo[1],lo[2]],[hi[0],lo[1],lo[2]],[hi[0],hi[1],lo[2]],[lo[0],hi[1],lo[2]],
                [lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],[hi[0],hi[1],hi[2]],[lo[0],hi[1],hi[2]]])
    faces=[(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(1,2,6),(1,6,5),
           (2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    text='ply\nformat ascii 1.0\nelement vertex 8\nproperty float x\nproperty float y\nproperty float z\nelement face 12\nproperty list uchar int vertex_indices\nend_header\n'
    text+=''.join(' '.join(map(str,p))+'\n' for p in v)
    text+=''.join('3 '+' '.join(map(str,f))+'\n' for f in faces)
    Path(path).write_text(text)

class SionnaWorld:
    def __init__(self,cfg,geometry):
        import sionna.rt as rt
        import mitsuba as mi
        self.rt,self.mi=rt,mi;self.cfg=cfg;self.geometry=geometry
        self.scene=rt.Scene();self.scene.frequency=cfg['propagation']['carrier_hz']
        self.scene.tx_array=rt.PlanarArray(num_rows=1,num_cols=1,pattern='iso',polarization='V')
        self.scene.rx_array=rt.PlanarArray(num_rows=1,num_cols=1,pattern='iso',polarization='V')
        material=rt.ITURadioMaterial(name='building_concrete',itu_type='concrete',thickness=1.0)
        asset=ROOT/'artifacts/scenes'/geometry.hash;asset.mkdir(parents=True,exist_ok=True)
        for i,b in enumerate(geometry.buildings):
            file=asset/f'building_{i}.ply'
            # Geometry-hashed immutable assets can be shared by parallel seed jobs.
            if not file.exists():write_box_ply(file,b)
            self.scene.add(rt.SceneObject(fname=str(file),name=f'building_{i}',radio_material=material))
        self.tx=rt.Transmitter('tx',cfg['scene']['initial_uav_xyz_m'])
        self.rx=rt.Receiver('rx',cfg['scene']['initial_ugv_xyz_m'])
        self.scene.add(self.tx);self.scene.add(self.rx)
        self.solver=rt.PathSolver(deterministic=True)
        self.material_document={'model':'Sionna ITU-R concrete','polarization':'V',
                                'relative_permittivity':material.relative_permittivity.numpy().tolist(),
                                'conductivity_s_per_m':material.conductivity.numpy().tolist(),
                                'thickness_m':1.,'variant':mi.variant()}
        self.material_hash=array_hash(np.array(self.material_document['relative_permittivity']),
                                      np.array(self.material_document['conductivity_s_per_m']))
        self.cache={}

    def paths(self,x):
        """Ray geometry and Fresnel coefficients, followed by fixed FOV/target rules."""
        x=np.asarray(x,dtype=np.float64);key=x.tobytes()
        if key in self.cache:return self.cache[key]
        self.tx.position=x[:3].tolist();self.rx.position=x[3:].tolist()
        e=self.cfg['propagation'];p=self.solver(self.scene,max_depth=1,los=True,specular_reflection=True,
            diffuse_reflection=False,refraction=False,diffraction=False,
            samples_per_src=e.get('rt_samples_per_src',10000),max_num_paths_per_src=128,
            synthetic_array=True,seed=e.get('rt_seed',43001))
        valid=p.valid.numpy().reshape(-1)
        tau=p.tau.numpy().reshape(-1).astype(np.float64)
        a=p.a[0].numpy().reshape(-1).astype(np.float64)+1j*p.a[1].numpy().reshape(-1).astype(np.float64)
        z=np.stack([20*np.log10(np.maximum(abs(a),1e-30)),tau,p.phi_r.numpy().reshape(-1),
                    p.theta_r.numpy().reshape(-1),p.phi_t.numpy().reshape(-1),p.theta_t.numpy().reshape(-1)],axis=-1)
        mask=valid & (z[:,0]>=e['signal_gain_floor_db']) & (tau>=e['tau_min_s']) & (tau<=e['tau_max_s'])
        mask &= (z[:,3]<=np.pi/2+1e-6) & (z[:,5]>=np.pi/2-1e-6)
        indices=np.flatnonzero(mask);indices=indices[np.argsort(-z[indices,0],kind='stable')][:e['signal_max_paths']]
        z=z[indices];a=a[indices]
        # Sionna a is the passband field coefficient (carrier phase is applied by cir()).
        # Restore the absolute carrier phase in float64, consistent with Eq.3 offsets.
        a=a*np.exp(-2j*np.pi*e['carrier_hz']*z[:,1])
        result=(z,a)
        # Bounded cache only of traced geometry, never training labels.
        if len(self.cache)>=32768:self.cache.pop(next(iter(self.cache)))
        self.cache[key]=result
        return result

    def reference(self,x):
        return extract_reference_set(self.paths(x)[0],self.cfg)

class SoundingOracle:
    def __init__(self,cfg,world):self.cfg=cfg;self.__world=world
    def observe(self,x,probe_key):
        z,a=self.__world.paths(x)
        h=channel_from_paths(z,a,self.cfg)
        return simulate_received(h,probe_key,self.cfg)

class Evaluator:
    def __init__(self,cfg,world,queries,weights):
        self.cfg=cfg;self.queries=queries;self.weights=weights
        self.__reference=[world.reference(x) for x in queries]
    def evaluate(self,predictions):
        from .metrics import parameter_rmse
        return parameter_rmse(predictions,self.__reference,self.weights,self.cfg)
    def save_private(self,path):
        lengths=np.array([len(p) for p in self.__reference]);offsets=np.r_[0,np.cumsum(lengths)]
        np.savez_compressed(path,queries=self.queries,weights=self.weights,offsets=offsets,
                            reference=np.concatenate(self.__reference) if offsets[-1] else np.empty((0,6)))
