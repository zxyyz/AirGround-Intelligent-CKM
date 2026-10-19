"""Central HDF5 with short process locks and atomic window commits."""
from common import *
import contextlib,fcntl,h5py,numpy as np

@contextlib.contextmanager
def locked(mode='a'):
    with (EXP/'records.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            with h5py.File(EXP/'records.h5',mode) as h:yield h
        finally:fcntl.flock(lock,fcntl.LOCK_UN)

def key(phase,pair,stream,r):return f'{phase}/{pair}/{"shared" if r==0 else stream}/w{r:02d}'
def has_window(phase,pair,stream,r):
    if not (EXP/'records.h5').exists():return False
    with locked('r') as h:return key(phase,pair,stream,r) in h and bool(h[key(phase,pair,stream,r)].attrs.get('committed',False))

def digest(arrays):
    out=hashlib.sha256()
    for name in sorted(arrays):
        a=np.asarray(arrays[name]);out.update(name.encode());out.update(str(a.shape).encode())
        if a.dtype.kind in 'OUS':out.update(json.dumps(a.tolist(),sort_keys=True,ensure_ascii=False).encode())
        else:out.update(a.tobytes())
    return out.hexdigest()

def put_window(phase,pair,stream,r,arrays,meta):
    name=key(phase,pair,stream,r);value=digest(arrays)
    with locked() as h:
        if name in h and h[name].attrs.get('committed',False):
            assert h[name].attrs['digest']==value;return value
        pending=name+'_pending'
        if pending in h:del h[pending]
        g=h.require_group(pending)
        for k,a in arrays.items():
            a=np.asarray(a)
            if a.dtype.kind in 'OUS':g.create_dataset(k,data=a.astype(object),dtype=h5py.string_dtype('utf-8'),compression='gzip')
            else:g.create_dataset(k,data=a,compression='gzip',compression_opts=1,shuffle=True)
        g.attrs['metadata']=json.dumps(meta,default=lambda a:a.tolist(),sort_keys=True)
        g.attrs['digest']=value;g.attrs['committed']=True;h.flush()
        if name in h:del h[name]
        h.move(pending,name);h.flush()
    return value

def get_window(phase,pair,stream,r,signals=False):
    with locked('r') as h:
        g=h[key(phase,pair,stream,r)];assert g.attrs['committed']
        a={k:v[:] for k,v in g.items() if signals or k!='S'}
        return a,json.loads(g.attrs['metadata']),g.attrs['digest']

def history(phase,pair,stream,r):
    blocks=[get_window(phase,pair,stream,i)[0] for i in range(r+1)]
    x=np.concatenate([a['x'] for a in blocks]);z=np.concatenate([a['labels'] for a in blocks]);k=np.concatenate([a['n_paths'] for a in blocks]);valid=np.concatenate([a['valid'] for a in blocks])
    return x,z,k,valid
