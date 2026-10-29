"""S=H P+N, LS, unit-norm horizontal UCA, column-major vectorization."""
import numpy as np
from .geometry import directions

def noise_variance(cfg):
    s=cfg['sounding']
    return s['pilot_energy_ep']*10**(s['noise_reference_gain_db']/10)/(s['nt']*s['nr']*10**(s['noise_reference_frobenius_snr_db']/10))

def array_response(a,n,cfg,receive=True):
    wavelength=cfg['propagation']['speed_of_light_mps']/cfg['propagation']['carrier_hz']
    radius=wavelength/(4*np.sin(np.pi/n));phi=2*np.pi*np.arange(n)/n
    offsets=radius*np.c_[np.cos(phi),np.sin(phi),np.zeros(n)]
    return np.exp((1j if receive else -1j)*2*np.pi/wavelength*(directions(a)@offsets.T))/np.sqrt(n)

def joint_atom(xi,cfg,q=None):
    s=cfg['sounding'];q=q or cfg['estimator']['frequency_window_q']
    ar=array_response(xi[1:3],s['nr'],cfg,True);at=array_response(xi[3:5],s['nt'],cfg,False)
    d=np.exp(-2j*np.pi*np.arange(q)*s['delta_f_hz']*xi[0])
    return np.kron(d,np.kron(at.conj(),ar))

def channel_from_paths(paths,coefficients,cfg):
    s=cfg['sounding'];h=np.zeros((s['n_frequencies'],s['nr'],s['nt']),complex)
    for z,alpha in zip(paths,coefficients):
        ar=array_response(z[2:4],s['nr'],cfg,True);at=array_response(z[4:6],s['nt'],cfg,False)
        d=np.exp(-2j*np.pi*np.arange(s['n_frequencies'])*s['delta_f_hz']*z[1])
        h+=alpha*d[:,None,None]*ar[None,:,None]*at.conj()[None,None,:]
    return h

def simulate_received(h,probe_key,cfg,noiseless=False):
    s=cfg['sounding'];pilot=np.eye(s['nt'])*np.sqrt(s['pilot_energy_ep'])
    signal=h@pilot
    if not noiseless:
        rng=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(list(probe_key))))
        signal=signal+np.sqrt(noise_variance(cfg)/2)*(rng.standard_normal(h.shape)+1j*rng.standard_normal(h.shape))
    return signal.astype(np.complex64),pilot

def estimate_ls_csi(signal,pilot,cfg):
    return signal@pilot.conj().T/cfg['sounding']['pilot_energy_ep']

def build_hankel_snapshots(h,q):
    f,nr,nt=h.shape
    return np.stack([h[s:s+q].swapaxes(-1,-2).reshape(-1) for s in range(f-q+1)],axis=-1)
