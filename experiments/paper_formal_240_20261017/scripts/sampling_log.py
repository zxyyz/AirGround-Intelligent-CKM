from common import *
import numpy as np,torch
from learning import replay_weights

def trace(path,model,poses,lens,steps,record_indices=None):
    spec=model.spec;replay=('replay' in spec or spec in ['Lreplay','Lcombined']);n=len(lens);batch=model.cfg['training']['logical_batch_size']
    weights=torch.as_tensor(replay_weights(poses,lens,model.geometry),dtype=torch.float64) if replay else None
    streams=[];hashes=[]
    for original in model.samplers:
        rng=torch.Generator();rng.set_state(original.get_state().cpu());indices=[]
        for _ in range(steps):indices.append((torch.multinomial(weights,batch,replacement=True,generator=rng) if replay else torch.randint(n,(batch,),generator=rng)).numpy())
        ids=np.array(indices,dtype=np.int64);hashes.append(hashlib.sha256(ids.tobytes()).hexdigest());streams.append(ids.astype(np.int32))
    np.savez_compressed(path,local_valid_row_indices=np.array(streams),source_record_indices=np.arange(n) if record_indices is None else record_indices,n_train=n,steps=steps,sample_sha256=np.array(hashes),replay=replay)
    return hashes

def reconstruct_fixed():
    import h5py
    from support import OLD,cfg_for,unpack
    from learning import Committee
    for dest in sorted((EXP/'runs/fixed_data').glob('*/*/*')):
        if not (dest/'complete.json').exists():continue
        world,flow,spec=dest.parts[-3:];cfg=cfg_for(world);cfg['runtime']['device']='cpu';m=Committee(cfg,int(flow),spec)
        with h5py.File(OLD/f'runs/v1_4/{world}/2401/C/records.h5') as h:poses=h['x'][:];lens=h['n_paths'][:]
        allowed=np.load(EXP/f'cache/fixed_train_{world}.npz')['indices']
        for j,(n,steps) in enumerate(zip([1800,3600,5400,7200],[1500,800,800,800])):
            ix=allowed[allowed<n]
            if j:
                prev=torch.load(dest/f'update_{j-1}.pt',map_location='cpu',weights_only=False)
                for r,s in zip(m.samplers,prev['committee']['samplers']):r.set_state(s.cpu())
            hashes=trace(dest/f'sample_indices_{j}.npz',m,poses[ix],lens[ix],steps,ix)
            logs=json.loads((dest/f'training_{j}.json').read_text())['models'];assert hashes==[r['sample_index_sha256'] for r in logs]
    save(EXP/'results/fixed_sampling_indices_verified.json',dict(epoch=time.time(),status='all completed fixed-data actual sample index hashes reproduced exactly from pre-update sampler states'))
if __name__=='__main__':reconstruct_fixed()
