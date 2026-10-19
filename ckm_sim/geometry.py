"""Public geometry only. The legacy PCG64DXSM district generator is retained."""
import numpy as np
from .config import array_hash

def angles(u):
    u = np.asarray(u)
    return np.stack([np.arctan2(u[...,1], u[...,0]),
                     np.arctan2(np.linalg.norm(u[...,:2], axis=-1), u[...,2])], -1)

def directions(a):
    a = np.asarray(a)
    return np.stack([np.sin(a[...,1])*np.cos(a[...,0]),
                     np.sin(a[...,1])*np.sin(a[...,0]), np.cos(a[...,1])], -1)

class GeometryMap:
    def __init__(self, cfg):
        self.cfg = cfg; s = cfg['scene']; self.s = s
        rng = np.random.Generator(np.random.PCG64DXSM(s['geometry_seed']))
        centers = [(x,y) for y in s['building_centers_axis_m'] for x in s['building_centers_axis_m']]
        heights = rng.uniform(*s['building_height_range_m'], len(centers))
        half = np.asarray(s['building_footprint_xy_m'])/2
        self.buildings = np.array([[x-half[0],y-half[1],x+half[0],y+half[1],h]
                                  for (x,y),h in zip(centers,heights)])
        roads = np.array(s['road_coordinates_m']); n = len(roads)
        self.nodes = np.array([(x,y) for y in roads for x in roads])
        self.edges = np.array([(n*j+i,n*j+i+1) for j in range(n) for i in range(n-1)] +
                              [(n*j+i,n*(j+1)+i) for i in range(n) for j in range(n-1)])
        self.hash = array_hash(self.buildings, self.nodes, self.edges)
        self.lengths = np.linalg.norm(self.nodes[self.edges[:,1]]-self.nodes[self.edges[:,0]], axis=1)
        self.node_dist = np.abs(self.nodes[:,None]-self.nodes[None]).sum(-1)
        self.faces = [(i,j,axis,plane,side) for i,b in enumerate(self.buildings)
                      for j,(axis,plane,side) in enumerate([(0,b[0],-1),(0,b[2],1),(1,b[1],-1),(1,b[3],1)])]

    def floor(self, xy):
        xy = np.asarray(xy)
        b = self.buildings
        d = np.maximum(np.maximum(b[:,:2]-xy[...,None,:], xy[...,None,:]-b[:,2:4]), 0)
        near = np.linalg.norm(d,axis=-1) <= self.s['horizontal_clearance_m']+1e-9
        return np.maximum(self.s['uav_base_height_m'],
                          np.max(np.where(near,b[:,4]+self.s['roof_clearance_m'],0),axis=-1))

    def air_safe(self, p):
        p = np.asarray(p)
        return ((p[...,:2]>=-1e-8).all(-1) & (p[...,:2]<=np.array(self.s['size_xy_m'])+1e-8).all(-1)
                & (p[...,2]>=self.floor(p[...,:2])-1e-8) & (p[...,2]<=self.s['uav_max_height_m']+1e-8))

    def road_attach(self, p):
        p = np.atleast_2d(p)[...,:2]
        a = self.nodes[self.edges[:,0]]; d = self.nodes[self.edges[:,1]]-a
        t = np.sum((p[:,None]-a)*d,axis=-1)/self.lengths**2
        tc = np.clip(t,0,1)
        err = np.linalg.norm(p[:,None]-a-tc[...,None]*d,axis=-1)
        ids = np.argmin(err,axis=1)
        if np.any(err[np.arange(len(p)),ids]>1e-6):
            raise ValueError('Off-road point')
        return ids, tc[np.arange(len(p)),ids]*self.lengths[ids]

    def road_distance(self, a, b):
        a,b = np.atleast_2d(a),np.atleast_2d(b)
        ia,sa=self.road_attach(a); ib,sb=self.road_attach(b)
        aa=self.edges[ia]; bb=self.edges[ib]
        da=np.stack([sa,self.lengths[ia]-sa],-1); db=np.stack([sb,self.lengths[ib]-sb],-1)
        out=np.full((len(a),len(b)),np.inf)
        for i in range(2):
            for j in range(2):
                out=np.minimum(out,da[:,i,None]+self.node_dist[aa[:,i,None],bb[None,:,j]]+db[None,:,j])
        return np.minimum(out,np.where(ia[:,None]==ib[None,:],abs(sa[:,None]-sb[None,:]),np.inf))

    def validate_joint_state(self,x):
        x=np.atleast_2d(x)
        if not self.air_safe(x[:,:3]).all(): raise ValueError('Unsafe UAV state')
        self.road_attach(x[:,3:])
        if not np.allclose(x[:,5],self.s['ugv_antenna_height_m']): raise ValueError('UGV height')
        return True

    def clear(self,a,b):
        """Exact line/open-box slab intersection; facade endpoints are permitted."""
        a,b=np.asarray(a),np.asarray(b); delta=b-a
        boxes=self.buildings
        lo=np.c_[boxes[:,:2],np.zeros(len(boxes))]+1e-7
        hi=np.c_[boxes[:,2:4],boxes[:,4]]-1e-7
        t0=np.zeros(len(boxes));t1=np.ones(len(boxes));possible=np.ones(len(boxes),bool)
        for j in range(3):
            if abs(delta[j])<1e-12:
                possible &= (a[j]>lo[:,j]) & (a[j]<hi[:,j])
            else:
                u=(lo[:,j]-a[j])/delta[j];v=(hi[:,j]-a[j])/delta[j]
                t0=np.maximum(t0,np.minimum(u,v));t1=np.minimum(t1,np.maximum(u,v))
        return not np.any(possible & (t0<t1))

    def candidate_paths(self,x):
        """Mirror-method paths: stable face ID, length, directions, valid flag."""
        u,g=np.asarray(x[:3]),np.asarray(x[3:]);out=[]
        d=g-u; length=np.linalg.norm(d)
        if length>0:out.append((-1,0,length,-d/length,d/length,self.clear(u,g)))
        for i,j,axis,plane,side in self.faces:
            box=self.buildings[i]
            if side*(u[axis]-plane)<=1e-8 or side*(g[axis]-plane)<=1e-8:continue
            image=u.copy();image[axis]=2*plane-u[axis];den=g[axis]-image[axis]
            if abs(den)<1e-12:continue
            t=(plane-image[axis])/den
            if not 0<t<1:continue
            p=image+t*(g-image);other=1-axis
            if not box[other]-1e-8<=p[other]<=box[other+2]+1e-8 or not 0<=p[2]<=box[4]:continue
            first=p-u;last=p-g;d1=np.linalg.norm(first);d2=np.linalg.norm(last)
            valid=self.clear(u,p) and self.clear(p,g)
            out.append((4*i+j,1,d1+d2,last/d2,first/d1,valid))
        return sorted(out,key=lambda r:(not r[5],r[2],r[0]))

    def candidate_path_features(self,x):
        n=self.cfg['model']['geometry_candidate_count'];c=self.cfg['propagation']['speed_of_light_mps']
        rows=np.zeros((n,10));paths=self.candidate_paths(x)[:n]
        for i,(_,order,length,a,d,valid) in enumerate(paths):
            rows[i]=[valid,order,length/c/self.cfg['propagation']['tau_max_s'],*a,*d,1]
        position=np.asarray(x)/np.array([*self.s['size_xy_m'],self.s['uav_max_height_m'],
                                        *self.s['size_xy_m'],self.s['uav_max_height_m']])
        return np.r_[position,rows.ravel()].astype(np.float32)

    def features(self,x):
        return np.stack([self.candidate_path_features(p) for p in np.atleast_2d(x)])

    def sample_safe_queries(self,count,seed):
        rng=np.random.Generator(np.random.PCG64DXSM(seed));air=[]
        while len(air)<count:
            p=rng.uniform([0,0,self.s['uav_base_height_m']],
                          [*self.s['size_xy_m'],self.s['uav_max_height_m']],(max(count,64),3))
            air.extend(p[self.air_safe(p)])
        air=np.array(air[:count]);ids=rng.choice(len(self.edges),count,p=self.lengths/self.lengths.sum())
        t=rng.uniform(size=count);ground=self.nodes[self.edges[ids,0]]+t[:,None]*(self.nodes[self.edges[ids,1]]-self.nodes[self.edges[ids,0]])
        return np.c_[air,ground,np.full(count,self.s['ugv_antenna_height_m'])],np.full(count,1/count)
