"""Unfiltered Sionna paths and auditable postfilters for isolated diagnostics."""
from common import *
import numpy as np
from ckm_sim.geometry import GeometryMap
from ckm_sim.propagation import write_box_ply

MODES={
 'R0':dict(max_depth=1,diffraction=False,diffuse_reflection=False,refraction=False),
 'R1':dict(max_depth=2,diffraction=False,diffuse_reflection=False,refraction=False),
 'R2':dict(max_depth=2,diffraction=True,diffuse_reflection=False,refraction=False),
 'R3':dict(max_depth=2,diffraction=False,diffuse_reflection=True,refraction=False),
 'T1':dict(max_depth=4,diffraction=False,diffuse_reflection=False,refraction=True),
}

def material_record(m):
    return dict(name=m.name,relative_permittivity=m.relative_permittivity.numpy().tolist(),conductivity_s_per_m=m.conductivity.numpy().tolist(),thickness_m=m.thickness.numpy().tolist(),scattering_coefficient=m.scattering_coefficient.numpy().tolist())

class AuditWorld:
    def __init__(self,scene_label='H0',variant='M0',mesh_manifest=None,scattering=0.0):
        import sionna.rt as rt
        import mitsuba as mi
        self.rt=rt;self.mi=mi;self.cfg=config(scene_label);self.variant=variant
        self.geometry=GeometryMap(self.cfg)
        if mesh_manifest is not None and mesh_manifest.get('mode')=='random_floors':
            self.geometry.buildings=np.array([b['bounds'] for b in mesh_manifest['buildings']])
            self.geometry.hash=mesh_manifest['geometry_hash']
        self.scene=rt.Scene();self.scene.frequency=3.5e9
        self.scene.tx_array=rt.PlanarArray(num_rows=1,num_cols=1,pattern='iso',polarization='V')
        self.scene.rx_array=rt.PlanarArray(num_rows=1,num_cols=1,pattern='iso',polarization='V')
        materials={}
        objects=[]
        if mesh_manifest is None:
            materials['concrete']=rt.ITURadioMaterial(name='audit_concrete',itu_type='concrete',thickness=1.,scattering_coefficient=scattering)
            dest=EXP/'assets'/f'boxes_{scene_label}';dest.mkdir(parents=True,exist_ok=True)
            for i,b in enumerate(self.geometry.buildings):
                path=dest/f'building_{i}.ply'
                if not path.exists():write_box_ply(path,b)
                objects.append(rt.SceneObject(fname=str(path),name=f'building_{i}',radio_material=materials['concrete']))
        else:
            for key,v in mesh_manifest['materials'].items():
                materials[key]=rt.ITURadioMaterial(name='audit_'+key,itu_type=v['itu_type'],thickness=v['thickness_m'],scattering_coefficient=v.get('scattering_coefficient',0.0))
            for v in mesh_manifest['meshes']:
                if v.get('enabled',True):objects.append(rt.SceneObject(fname=str(EXP/v['path']),name=v['name'],radio_material=materials[v['material']]))
        self.scene.add(objects)
        self.tx=rt.Transmitter('tx',self.cfg['scene']['initial_uav_xyz_m']);self.rx=rt.Receiver('rx',self.cfg['scene']['initial_ugv_xyz_m'])
        self.scene.add([self.tx,self.rx]);self.solver=rt.PathSolver(deterministic=True)
        self.object_map={int(o.object_id):dict(name=o.name,material=o.radio_material.name) for o in objects}
        self.materials={k:material_record(v) for k,v in materials.items()}
        self.manifest=dict(scene=scene_label,variant=variant,geometry_hash=self.geometry.hash,frequency_hz=3.5e9,materials=self.materials,objects=self.object_map,mitsuba_variant=mi.variant(),mesh_manifest=mesh_manifest)

    def trace(self,x,mode='R0',samples=10000,max_paths=128,seed=43001):
        check_deadline()
        self.tx.position=x[:3];self.rx.position=x[3:]
        params=dict(MODES[mode]);start=time.perf_counter()
        p=self.solver(self.scene,**params,los=True,specular_reflection=True,synthetic_array=True,samples_per_src=samples,max_num_paths_per_src=max_paths,seed=seed)
        valid=p.valid.numpy().reshape(-1)
        n=len(valid)
        tau=p.tau.numpy().reshape(-1).astype(float)
        a=p.a[0].numpy().reshape(-1).astype(float)+1j*p.a[1].numpy().reshape(-1).astype(float)
        z=np.stack([20*np.log10(np.maximum(abs(a),1e-30)),tau,p.phi_r.numpy().reshape(-1),p.theta_r.numpy().reshape(-1),p.phi_t.numpy().reshape(-1),p.theta_t.numpy().reshape(-1)],axis=-1)
        inter=p.interactions.numpy();depth=inter.shape[0]
        inter=inter.reshape(depth,n)
        objects=p.objects.numpy().reshape(depth,n)
        primitives=p.primitives.numpy().reshape(depth,n)
        vertices=p.vertices.numpy().reshape(depth,n,3)
        ids=np.flatnonzero(valid & np.isfinite(z).all(axis=1) & (tau>=0))
        ids=ids[np.argsort(-z[ids,0],kind='stable')]
        paths=[]
        for i in ids:
            hits=[]
            for d in range(depth):
                kind=int(inter[d,i])
                if kind:
                    obj=int(objects[d,i]);v=vertices[d,i]
                    hits.append(dict(type=kind,object_id=obj,primitive=int(primitives[d,i]),vertex=v.tolist(),**self.object_map.get(obj,dict(name='unmapped',material='unmapped'))))
            paths.append(dict(z=z[i].tolist(),coefficient=[float(a[i].real),float(a[i].imag)],hits=hits,order=len(hits),kind='los' if not hits else '+'.join(str(h['type']) for h in hits)))
        return dict(position=list(x),mode=mode,variant=self.variant,solver=dict(**params,samples_per_src=samples,max_num_paths_per_src=max_paths,seed=seed),paths=paths,elapsed_s=time.perf_counter()-start,array_shapes=dict(valid=list(p.valid.shape),interactions=list(p.interactions.shape),vertices=list(p.vertices.shape)),raw_valid_count=int(valid.sum()))

def filtered(rec,cfg=None,threshold=-110.,fov=True,cap=3):
    cfg=config() if cfg is None else cfg
    e=cfg['propagation'];out=[]
    for i,p in enumerate(rec['paths']):
        z=p['z']
        if z[1]<e['tau_min_s'] or z[1]>e['tau_max_s']:continue
        if fov and (z[3]>np.pi/2+1e-6 or z[5]<np.pi/2-1e-6):continue
        if z[0]>=threshold:out.append(i)
    return out if cap is None else out[:cap]

def summarize(rec,sample):
    raw=rec['paths'];legal=filtered(rec,threshold=-600,fov=True,cap=None)
    delay=filtered(rec,threshold=-600,fov=False,cap=None)
    sig=filtered(rec,threshold=-140,fov=True,cap=6)
    target=[i for i in sig if raw[i]['z'][0]>=-110][:3]
    selected=[raw[i] for i in target]
    return dict(scene=sample['scene'],sample_id=sample['sample_id'],stratum=sample['stratum'],variant=rec['variant'],mode=rec['mode'],geometry_candidates=sample.get('geometry_candidates',''),raw_paths=len(raw),delay_legal=len(delay),fov_delay_legal=len(legal),signal_paths=len(sig),ckm_k=len(target),raw_los=sum(p['order']==0 for p in raw),raw_specular2=sum(p['kind']=='1+1' for p in raw),raw_diffraction=sum(any(h['type']==8 for h in p['hits']) for p in raw),ckm_specular2=sum(p['kind']=='1+1' for p in selected),ckm_diffraction=sum(any(h['type']==8 for h in p['hits']) for p in selected),ckm_glass=sum(any(h['material']=='audit_glass' for h in p['hits']) for p in selected),raw_glass=sum(any(h['material']=='audit_glass' for h in p['hits']) for p in raw),raw_ground=sum(any(h['material'] in ('audit_asphalt','audit_ground') for h in p['hits']) for p in raw),ckm_ground=sum(any(h['material'] in ('audit_asphalt','audit_ground') for h in p['hits']) for p in selected),best_gain_db=raw[0]['z'][0] if raw else '',elapsed_s=rec['elapsed_s'])

def cache_trace(world,sample,mode='R0',tag=None,**kw):
    tag=tag or f'{world.variant}_{mode}'
    path=EXP/'cache'/tag/(sample['sample_id']+'.json')
    if path.exists():return json.loads(path.read_text())
    rec=world.trace(sample['position'],mode,**kw)
    rec.update(sample_id=sample['sample_id'],scene=sample['scene'],stratum=sample['stratum'])
    save(path,rec)
    return rec

