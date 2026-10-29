"""Frequency-smoothed 5-D joint MUSIC, observation-only rank, joint LS."""
import itertools
import time
import numpy as np
import torch
from .config import ROOT, write_json, scientific_hash
from .geometry import directions
from .sounding import (noise_variance, estimate_ls_csi, build_hankel_snapshots,
                       array_response, joint_atom)

def signal_subspace(y,threshold,max_rank):
    # The Gram eigendecomposition is algebraically the thin SVD; no full D x D covariance.
    # Signal/noise eigenvalues can differ by 1e6. Double precision avoids
    # cancellation that otherwise changes the estimated order under batching.
    working=y.to(torch.complex128)
    gram=working.mH@working/y.shape[-1]
    values,v=torch.linalg.eigh(gram)
    rank=min(int((values>threshold).sum().item()),max_rank)
    if not rank:return y[:,:0],values,0
    vals=values[-rank:];vec=v[:,-rank:]
    us=(working@vec)/torch.sqrt(torch.clamp(vals*y.shape[-1],min=1e-30))[None,:]
    # A second thin QR controls float32 orthogonality without changing the span.
    us=torch.linalg.qr(us,mode='reduced')[0]
    return us.to(y.dtype),values,rank

def music_score(us,atoms,floor=1e-7):
    atoms=atoms/torch.linalg.vector_norm(atoms,dim=0)
    return 1/torch.clamp(1-(us.mH@atoms).abs().square().sum(0),min=floor)

class RFEstimator:
    def __init__(self,cfg,calibration=None):
        self.cfg=cfg;self.s=cfg['sounding'];self.e=cfg['estimator']
        self.device=cfg['runtime'].get('device','cuda:0')
        if self.device.startswith('auto'):self.device='cuda:0' if torch.cuda.is_available() else 'cpu'
        self.dtype=torch.complex128 if cfg['runtime']['rf_dtype']=='complex128' else torch.complex64
        self.variance=noise_variance(cfg)/self.s['pilot_energy_ep']
        self.threshold=None if calibration is None else calibration['normalized_rank_threshold']
        self.q=self.e['frequency_window_q']
        nd,na,nz=(self.e[k] for k in ['coarse_delay_bins','coarse_azimuth_bins','coarse_zenith_bins'])
        self.nd,self.na,self.nz=nd,na,nz
        self.delay=np.linspace(cfg['propagation']['tau_min_s'],cfg['propagation']['tau_max_s'],nd)
        az=np.arange(na)*2*np.pi/na-np.pi
        za=np.linspace(0,np.pi/2,nz);zd=np.linspace(np.pi/2,np.pi,nz)
        self.rxangles=np.array(list(itertools.product(az,za)))
        self.txangles=np.array(list(itertools.product(az,zd)))
        self.ar=self.tensor(array_response(self.rxangles,self.s['nr'],cfg,True))
        self.at=self.tensor(array_response(self.txangles,self.s['nt'],cfg,False))
        self.freq=self.tensor(np.exp(-2j*np.pi*self.delay[:,None]*np.arange(self.q)*self.s['delta_f_hz'])/np.sqrt(self.q))
        self.step=np.array([(self.delay[-1]-self.delay[0])/(nd-1),2*np.pi/na,np.pi/(2*(nz-1)),2*np.pi/na,np.pi/(2*(nz-1))])

    def tensor(self,value):return torch.as_tensor(value,dtype=self.dtype,device=self.device)

    def snapshots(self,h):
        return self.tensor(build_hankel_snapshots(h,self.q)/np.sqrt(self.variance))

    def calibrate(self,out=None):
        e=self.e;rng=np.random.Generator(np.random.PCG64DXSM(e['noise_calibration_seed']))
        shape=(self.s['n_frequencies'],self.s['nr'],self.s['nt']);maxima=[]
        start=time.perf_counter()
        for _ in range(e['noise_calibration_trials']):
            h=np.sqrt(self.variance/2)*(rng.standard_normal(shape)+1j*rng.standard_normal(shape))
            y=self.snapshots(h).to(torch.complex128);maxima.append(float(torch.linalg.eigvalsh(y.mH@y/y.shape[1])[-1].cpu()))
        self.threshold=float(np.quantile(maxima,e['noise_false_alarm_quantile']))
        held=[]
        for _ in range(100):
            h=np.sqrt(self.variance/2)*(rng.standard_normal(shape)+1j*rng.standard_normal(shape))
            y=self.snapshots(h).to(torch.complex128);held.append(float(torch.linalg.eigvalsh(y.mH@y/y.shape[1])[-1].cpu()))
        result={'normalized_rank_threshold':self.threshold,'noise_variance_h':self.variance,
                'seed':e['noise_calibration_seed'],'trials':len(maxima),'quantile':e['noise_false_alarm_quantile'],
                'heldout_trials':len(held),'heldout_false_alarm_rate':float(np.mean(np.array(held)>self.threshold)),
                'maximum_noise_eigenvalues':maxima,'heldout_maximum_eigenvalues':held,
                'rf_dtype':str(self.dtype),'device':self.device,'elapsed_s':time.perf_counter()-start,
                'shape':[self.q*self.s['nt']*self.s['nr'],self.s['n_frequencies']-self.q+1]}
        if out:write_json(out,result)
        return result

    def coarse_peaks(self,us):
        u=us.reshape(self.q,self.s['nt'],self.s['nr'],-1)
        # Each tensor contraction preserves the associated delay/AoA/AoD tuple.
        tmp=torch.einsum('fq,qtrk->ftrk',self.freq,u.conj())
        tmp=torch.einsum('ar,ftrk->fatk',self.ar,tmp)
        projection=torch.einsum('dt,fatk->fadk',self.at.conj(),tmp)
        score=1/torch.clamp(1-projection.abs().square().sum(-1),min=self.e['spectrum_floor'])
        score=score.reshape(self.nd,self.na,self.nz,self.na,self.nz)
        maximum=torch.ones_like(score,dtype=torch.bool)
        for axis in range(5):
            for step in [-1,1]:
                neighbor=torch.roll(score,step,axis)
                if axis in [0,2,4]:
                    sl=[slice(None)]*5;sl[axis]=0 if step==1 else -1
                    neighbor[tuple(sl)]=-torch.inf
                maximum &= score>=neighbor
        # Deduplicate all azimuths at each physical pole before top-k.
        maximum[:,1:,0,:,:]=False;maximum[:,:,:,1:,-1]=False
        flat=torch.where(maximum,score,-torch.inf).flatten()
        k=min(self.e['refine_candidates'],int(maximum.sum().item()))
        indices=torch.topk(flat,k).indices.cpu().numpy()
        f,a,z,d,w=np.unravel_index(indices,score.shape)
        return np.c_[self.delay[f],self.rxangles[a*self.nz+z],self.txangles[d*self.nz+w]]

    def refine(self,us,xi):
        u=us.reshape(self.q,self.s['nt'],self.s['nr'],-1)
        local=np.array(list(itertools.product([-1,0,1],repeat=2)))
        step=self.step.copy()
        for _ in range(self.e['refine_levels']):
            delay=np.clip(xi[:,0,None]+step[0]*np.array([-1,0,1]),self.delay[0],self.delay[-1])
            aa=xi[:,None,1:3]+local[None]*step[None,None,1:3]
            dd=xi[:,None,3:5]+local[None]*step[None,None,3:5]
            aa[...,0]=(aa[...,0]+np.pi)%(2*np.pi)-np.pi;dd[...,0]=(dd[...,0]+np.pi)%(2*np.pi)-np.pi
            aa[...,1]=np.clip(aa[...,1],0,np.pi/2);dd[...,1]=np.clip(dd[...,1],np.pi/2,np.pi)
            freq=self.tensor(np.exp(-2j*np.pi*delay[...,None]*np.arange(self.q)*self.s['delta_f_hz'])/np.sqrt(self.q))
            ar=self.tensor(array_response(aa,self.s['nr'],self.cfg,True))
            at=self.tensor(array_response(dd,self.s['nt'],self.cfg,False))
            tmp=torch.einsum('cfq,qtrk->cftrk',freq,u.conj())
            tmp=torch.einsum('car,cftrk->cfatk',ar,tmp)
            p=torch.einsum('cdt,cfatk->cfadk',at.conj(),tmp)
            norm=p.abs().square().sum(-1)
            flat=norm.flatten(1);idx=flat.argmax(1).cpu().numpy();f,a,d=np.unravel_index(idx,(3,9,9));row=np.arange(len(xi))
            xi=np.c_[delay[row,f],aa[row,a],dd[row,d]]
            step*=self.e['refine_step_ratio']
        atoms=self.tensor(np.stack([joint_atom(x,self.cfg) for x in xi],axis=1))
        scores=music_score(us,atoms,self.e['spectrum_floor']).cpu().numpy()
        return xi,scores

    def fit_path_gains(self,h,xi):
        atoms=np.stack([joint_atom(x,self.cfg,self.s['n_frequencies']) for x in xi],axis=1)
        target=h.swapaxes(-1,-2).reshape(-1)
        alpha,_,rank,singular=np.linalg.lstsq(atoms,target,rcond=self.e['numerical_rcond'])
        residual=float(np.linalg.norm(atoms@alpha-target)/max(np.linalg.norm(target),1e-30))
        condition=float(singular[0]/max(singular[-1],1e-30))
        paths=np.c_[20*np.log10(np.maximum(abs(alpha),1e-30)),xi]
        indices=np.argsort(-paths[:,0],kind='stable');paths=paths[indices]
        t=self.cfg['ckm_target'];keep=(paths[:,0]>=t['gain_min_db']) & (paths[:,1]<=t['max_delay_s'])
        return paths[keep][:t['max_paths']],{'residual':residual,'condition':condition,'response_rank':int(rank)}

    def estimate(self,signal,pilots,calibration=None):
        start=time.perf_counter()
        if calibration:self.threshold=calibration['normalized_rank_threshold']
        if self.threshold is None:raise RuntimeError('Noise calibration required')
        h=estimate_ls_csi(signal,pilots,self.cfg);y=self.snapshots(h)
        us,values,rank=signal_subspace(y,self.threshold,self.e['max_fit_paths'])
        quality={'valid':True,'order':rank,'rank_threshold':self.threshold,'reason':'ok',
                 'maximum_eigenvalue':float(values[-1].cpu())}
        if rank==0:
            quality['elapsed_s']=time.perf_counter()-start
            return np.empty((0,6)),quality
        xi=self.coarse_peaks(us);xi,scores=self.refine(us,xi)
        chosen=[]
        for i in np.argsort(-scores,kind='stable'):
            candidate=xi[i];duplicate=False
            for existing in chosen:
                aa=np.linalg.norm(directions(candidate[1:3])-directions(existing[1:3]))
                dd=np.linalg.norm(directions(candidate[3:5])-directions(existing[3:5]))
                tol=2*np.sin(np.deg2rad(self.e['peak_direction_separation_deg'])/2)
                if abs(candidate[0]-existing[0])<self.e['peak_delay_separation_s'] and aa<tol and dd<tol:
                    duplicate=True;break
            if not duplicate:chosen.append(candidate)
            if len(chosen)>=rank:break
        if not chosen:
            quality.update(valid=False,reason='no_spectrum_peaks',elapsed_s=time.perf_counter()-start)
            return np.empty((0,6)),quality
        paths,fit=self.fit_path_gains(h,np.array(chosen));quality.update(fit)
        quality.update(peak_count=len(chosen),coarse_points=self.nd*(self.na*self.nz)**2,
                       refinement_points=len(xi)*3**5*self.e['refine_levels'],elapsed_s=time.perf_counter()-start)
        return paths,quality

    def estimate_many(self,signals,pilots):
        """Identical per-frame objective, batched Gram/projector and 5-D search."""
        if self.cfg['runtime'].get('receiver_batch_size',1)==1:
            result=[self.estimate(signal,pilots) for signal in signals]
            return [p for p,q in result],[q for p,q in result]
        start=time.perf_counter();h=estimate_ls_csi(signals,pilots,self.cfg)
        y=self.tensor(np.stack([build_hankel_snapshots(frame,self.q) for frame in h])/np.sqrt(self.variance))
        working=y.to(torch.complex128)
        values,v=torch.linalg.eigh(working.mH@working/y.shape[-1])
        ranks=(values>self.threshold).sum(-1).clamp_max(self.e['max_fit_paths']).cpu().numpy()
        paths=[np.empty((0,6)) for _ in signals]
        quality=[{'valid':True,'order':int(rank),'rank_threshold':self.threshold,'reason':'ok',
                  'maximum_eigenvalue':float(values[i,-1].cpu())} for i,rank in enumerate(ranks)]
        for rank in range(1,self.e['max_fit_paths']+1):
            ids=np.flatnonzero(ranks==rank)
            if not len(ids):continue
            us=(working[ids]@v[ids,:,-rank:])/torch.sqrt(values[ids,-rank:]*y.shape[-1])[:,None,:]
            us=torch.linalg.qr(us,mode='reduced')[0].to(self.dtype)
            u=us.reshape(len(ids),self.q,self.s['nt'],self.s['nr'],rank)
            tmp=torch.einsum('fq,bqtrk->bftrk',self.freq,u.conj())
            tmp=torch.einsum('ar,bftrk->bfatk',self.ar,tmp)
            projection=torch.einsum('dt,bfatk->bfadk',self.at.conj(),tmp)
            score=(1/torch.clamp(1-projection.abs().square().sum(-1),min=self.e['spectrum_floor'])).reshape(len(ids),self.nd,self.na,self.nz,self.na,self.nz)
            maximum=torch.ones_like(score,dtype=torch.bool)
            for axis in range(1,6):
                for direction in [-1,1]:
                    neighbor=torch.roll(score,direction,axis)
                    if axis in [1,3,5]:
                        sl=[slice(None)]*6;sl[axis]=0 if direction==1 else -1;neighbor[tuple(sl)]=-torch.inf
                    maximum &= score>=neighbor
            maximum[:,:,1:,0,:,:]=False;maximum[:,:,:,:,1:,-1]=False
            flat=torch.where(maximum,score,-torch.inf).flatten(1)
            count=min(self.e['refine_candidates'],int(maximum.flatten(1).sum(-1).min().cpu()))
            indices=torch.topk(flat,count,dim=1).indices.cpu().numpy()
            f,a,z,d,w=np.unravel_index(indices,score.shape[1:])
            xi=np.concatenate([self.delay[f][...,None],self.rxangles[a*self.nz+z],self.txangles[d*self.nz+w]],-1)
            step=self.step.copy();local=np.array(list(itertools.product([-1,0,1],repeat=2)))
            for _ in range(self.e['refine_levels']):
                delay=np.clip(xi[...,0,None]+step[0]*np.array([-1,0,1]),self.delay[0],self.delay[-1])
                aa=xi[...,None,1:3]+local[None,None]*step[None,None,None,1:3]
                dd=xi[...,None,3:5]+local[None,None]*step[None,None,None,3:5]
                aa[...,0]=(aa[...,0]+np.pi)%(2*np.pi)-np.pi;dd[...,0]=(dd[...,0]+np.pi)%(2*np.pi)-np.pi
                aa[...,1]=np.clip(aa[...,1],0,np.pi/2);dd[...,1]=np.clip(dd[...,1],np.pi/2,np.pi)
                freq=self.tensor(np.exp(-2j*np.pi*delay[...,None]*np.arange(self.q)*self.s['delta_f_hz'])/np.sqrt(self.q))
                ar=self.tensor(array_response(aa,self.s['nr'],self.cfg,True));at=self.tensor(array_response(dd,self.s['nt'],self.cfg,False))
                tmp=torch.einsum('bcfq,bqtrk->bcftrk',freq,u.conj())
                tmp=torch.einsum('bcar,bcftrk->bcfatk',ar,tmp)
                projection=torch.einsum('bcdt,bcfatk->bcfadk',at.conj(),tmp)
                norm=projection.abs().square().sum(-1)
                best=norm.flatten(2).argmax(-1).cpu().numpy();fi,ai,di=np.unravel_index(best,(3,9,9))
                batch=np.arange(len(ids))[:,None];candidate=np.arange(count)[None,:]
                xi=np.concatenate([delay[batch,candidate,fi][...,None],aa[batch,candidate,ai],dd[batch,candidate,di]],-1)
                step*=self.e['refine_step_ratio']
            # Evaluate final tuples with the same factorized projector, not the true paths.
            freq=self.tensor(np.exp(-2j*np.pi*xi[...,0,None]*np.arange(self.q)*self.s['delta_f_hz'])/np.sqrt(self.q))
            ar=self.tensor(array_response(xi[...,1:3],self.s['nr'],self.cfg,True));at=self.tensor(array_response(xi[...,3:5],self.s['nt'],self.cfg,False))
            tmp=torch.einsum('bcq,bqtrk->bctrk',freq,u.conj());tmp=torch.einsum('bcr,bctrk->bctk',ar,tmp)
            norm=torch.einsum('bct,bctk->bck',at.conj(),tmp).abs().square().sum(-1).cpu().numpy()
            for bi,index in enumerate(ids):
                chosen=[]
                for ci in np.argsort(-norm[bi],kind='stable'):
                    candidate=xi[bi,ci];duplicate=False
                    for existing in chosen:
                        aa=np.linalg.norm(directions(candidate[1:3])-directions(existing[1:3]))
                        dd=np.linalg.norm(directions(candidate[3:5])-directions(existing[3:5]))
                        tol=2*np.sin(np.deg2rad(self.e['peak_direction_separation_deg'])/2)
                        if abs(candidate[0]-existing[0])<self.e['peak_delay_separation_s'] and aa<tol and dd<tol:duplicate=True;break
                    if not duplicate:chosen.append(candidate)
                    if len(chosen)>=rank:break
                if not chosen:quality[index].update(valid=False,reason='no_spectrum_peaks');continue
                paths[index],fit=self.fit_path_gains(h[index],np.array(chosen));quality[index].update(fit)
                quality[index].update(peak_count=len(chosen),coarse_points=self.nd*(self.na*self.nz)**2,
                                      refinement_points=count*3**5*self.e['refine_levels'])
        elapsed=time.perf_counter()-start
        for q in quality:q.update(batch_size=len(signals),elapsed_s=elapsed/len(signals))
        return paths,quality
