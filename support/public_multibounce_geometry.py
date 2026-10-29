"""PUBLIC geometry only: finite facade images, no RT/material/reference imports.

Tx -> face1 -> face2 -> Rx. Unfold Tx over face1 then face2;
intersect backwards with face2, then with face1. Every vertex is bounded,
reflection law/outward sides checked, all physical segments tested for boxes.
"""
import numpy as np,time
from ckm_sim.geometry import GeometryMap
D=40
FIELDS=['valid','order_half','tau_fraction','rx_x','rx_y','rx_z','tx_x','tx_y','tx_z','mask','face1_id','face2_id','n1_x','n1_y','n1_z','n2_x','n2_y','n2_z','p1_x','p1_y','p1_z','p2_x','p2_y','p2_z','segment0','segment1','segment2','cos_in1','cos_out1','cos_in2','cos_out2','finite1','finite2','law1','law2','clear0','clear1','clear2','is_los','is_double']

def canonical_vectors(rx,tx):
    rx=np.array(rx,copy=True);tx=np.array(tx,copy=True);rx[...,2]=abs(rx[...,2]);tx[...,2]=-abs(tx[...,2]);return rx,tx

class PublicGeometry:
    def __init__(self,cfg,geometry=None):
        self.cfg=cfg;self.g=geometry or GeometryMap(cfg);self.faces=np.array(self.g.faces);self.n=len(self.faces);self.axis=self.faces[:,2].astype(int);self.plane=self.faces[:,3];self.side=self.faces[:,4];self.bi=self.faces[:,0].astype(int)
        self.normals=np.zeros((self.n,3));self.normals[np.arange(self.n),self.axis]=self.side
        self.pairs=np.array([(i,j) for i in range(self.n) for j in range(self.n) if i!=j]);self.scale=np.array([240,240,120.])
    def g0(self,x):return self.g.candidate_path_features(x)
    def clear_many(self,a,b):
        a,b=np.broadcast_arrays(np.atleast_2d(a),np.atleast_2d(b));delta=b-a;boxes=self.g.buildings;lo=np.c_[boxes[:,:2],np.zeros(len(boxes))]+1e-7;hi=np.c_[boxes[:,2:4],boxes[:,4]]-1e-7
        t0=np.zeros((len(a),len(boxes)));t1=np.ones_like(t0);possible=np.ones_like(t0,dtype=bool)
        for j in range(3):
            par=abs(delta[:,j])<1e-12;den=np.where(par,1.,delta[:,j]);u=(lo[None,:,j]-a[:,None,j])/den[:,None];v=(hi[None,:,j]-a[:,None,j])/den[:,None]
            t0=np.maximum(t0,np.where(par[:,None],-np.inf,np.minimum(u,v)));t1=np.minimum(t1,np.where(par[:,None],np.inf,np.maximum(u,v)));possible &= ~par[:,None]|((a[:,None,j]>lo[None,:,j])&(a[:,None,j]<hi[None,:,j]))
        return ~np.any(possible&(t0<t1),axis=1)
    def finite(self,p,ids):
        box=self.g.buildings[self.bi[ids]];other=1-self.axis[ids];ii=np.arange(len(ids));v=p[ii,other]
        return (v>=box[ii,other]-1e-8)&(v<=box[ii,other+2]+1e-8)&(p[:,2]>=-1e-8)&(p[:,2]<=box[:,4]+1e-8)
    def double_vertices(self,x):
        u,g=np.array(x[:3]),np.array(x[3:]);f1,f2=self.pairs.T;ax1,ax2=self.axis[f1],self.axis[f2];ii=np.arange(len(f1));im1=np.tile(u,(len(f1),1));im1[ii,ax1]=2*self.plane[f1]-im1[ii,ax1];im2=im1.copy();im2[ii,ax2]=2*self.plane[f2]-im2[ii,ax2]
        den2=g[ax2]-im2[ii,ax2];t2=np.divide(self.plane[f2]-im2[ii,ax2],den2,out=np.full(len(f1),np.nan),where=abs(den2)>1e-12);p2=im2+t2[:,None]*(g-im2)
        den1=p2[ii,ax1]-im1[ii,ax1];t1=np.divide(self.plane[f1]-im1[ii,ax1],den1,out=np.full(len(f1),np.nan),where=abs(den1)>1e-12);p1=im1+t1[:,None]*(p2-im1)
        mask=(t1>1e-9)&(t1<1-1e-9)&(t2>1e-9)&(t2<1-1e-9)&self.finite(p1,f1)&self.finite(p2,f2)
        f1,f2,p1,p2=f1[mask],f2[mask],p1[mask],p2[mask]
        return f1,f2,p1,p2
    def make(self,x,order,ids,points):
        u,g=x[:3],x[3:];p=np.array([u,*points,g]);seg=np.diff(p,axis=0);lengths=np.linalg.norm(seg,axis=-1)
        if (lengths<1e-7).any():return None
        v=seg/lengths[:,None];law=[];cos=[]
        for j,f in enumerate(ids):
            n=self.normals[f];incoming=v[j];outgoing=v[j+1];ok=(incoming@n<-1e-8 and outgoing@n>1e-8 and np.linalg.norm(outgoing-(incoming-2*(incoming@n)*n))<1e-6);law.append(ok);cos.extend([-incoming@n,outgoing@n])
        self_cross=False
        if order==2:
            mat=np.stack([seg[0],-seg[2]],axis=-1)
            ts=np.linalg.lstsq(mat,p[2]-p[0],rcond=None)[0]
            self_cross=bool(np.linalg.norm(mat@ts-(p[2]-p[0]))<1e-7 and np.all(ts>1e-8) and np.all(ts<1-1e-8))
        clear=self.clear_many(p[:-1],p[1:]);valid=all(law) and clear.all() and not self_cross;rx,tx=canonical_vectors(-v[-1],v[0]);total=lengths.sum();row=np.zeros(D);row[:10]=[valid,order/2,total/self.cfg['propagation']['speed_of_light_mps']/self.cfg['propagation']['tau_max_s'],*rx,*tx,1]
        for j,f in enumerate(ids):row[10+j]=(f+1)/self.n;row[12+j*3:15+j*3]=self.normals[f];row[18+j*3:21+j*3]=points[j]/self.scale
        row[24:24+len(lengths)]=lengths/600
        row[27:27+len(cos)]=cos
        row[31:40]=[1 if order>=1 else 0,1 if order==2 else 0,law[0] if order>=1 else 0,law[1] if order==2 else 0,*list(clear)+[0]*(3-len(clear)),order==0,order==2]
        return dict(order=order,ids=tuple(ids),length=float(total),valid=bool(valid),row=row,vertices=p)
    def candidates(self,x,level=2):
        x=np.asarray(x);out=[];base=self.g.candidate_paths(x)
        for face,order,length,a,d,valid in base:
            points=[]
            if order:
                ax=int(self.axis[face]);im=x[:3].copy();im[ax]=2*self.plane[face]-im[ax];t=(self.plane[face]-im[ax])/(x[3+ax]-im[ax]);points=[im+t*(x[3:]-im)]
            rec=self.make(x,order,[] if order==0 else [face],points)
            if rec:out.append(rec)
        raw_double=self.n*(self.n-1);finite_double=0
        if level==2:
            f1,f2,p1,p2=self.double_vertices(x);finite_double=len(f1)
            for a,b,c,d in zip(f1,f2,p1,p2):
                rec=self.make(x,2,[a,b],[c,d])
                if rec:out.append(rec)
        out.sort(key=lambda r:(not r['valid'],r['length'],r['ids']))
        return out,dict(enumerated=1+self.n+(raw_double if level==2 else 0),finite=len(out),finite_double=finite_double,valid=sum(r['valid'] for r in out),valid_double=sum(r['valid'] and r['order']==2 for r in out))
    def features(self,x,level=2,capacity=64):
        rows=[];stats=[]
        for xx in np.atleast_2d(x):
            t=time.perf_counter();cand,stat=self.candidates(xx,level);f=np.zeros((capacity,D),np.float32)
            for j,c in enumerate(cand[:capacity]):f[j]=c['row']
            rows.append(np.r_[xx/np.r_[self.scale,self.scale],f.ravel()]);stats.append(dict(**stat,retained=min(capacity,len(cand)),valid_retained=sum(c['valid'] for c in cand[:capacity]),construction_s=time.perf_counter()-t))
        return np.array(rows,np.float32),stats
