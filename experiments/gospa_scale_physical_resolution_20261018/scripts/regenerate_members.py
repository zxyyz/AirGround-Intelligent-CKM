"""Read-only frozen model inference in an isolated legacy import namespace.

Only decoded full-query member sets are saved in the NEW experiment directory.
No RT/RF/trajectory/training functions are called.
"""
from pathlib import Path
import sys,os,json,time,hashlib
import numpy as np,torch

EXP=Path(__file__).resolve().parents[1];ROOT=EXP.parents[1];OLD=ROOT/'experiments/paper_formal_240_20261017'
sys.path.insert(0,str(OLD/'scripts'));sys.path.insert(0,str(ROOT))
from formal_models import make_model
from decoding import raw_predict,decode

def run():
    start=time.perf_counter();torch.set_num_threads(2);cfg=json.loads((OLD/'configs/effective_config.json').read_text());features=np.load(OLD/'cache/features_formal.npz');evidence=[]
    assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
    for pair in range(5101,5111):
        for branch in ['I-D','I-G','C-G','P-G']:
            variant='D' if branch=='I-D' else 'M3';m=make_model(cfg,pair,variant);path=OLD/f'runs/formal/{pair}/{branch}/checkpoints/update_07.pt';checkpoint=torch.load(path,map_location=m.device,weights_only=False);m.load_state_dict(checkpoint['committee'])
            z,l=raw_predict(m,features[variant]);saved=np.load(OLD/f'runs/formal/{pair}/{branch}/evaluation_07.npz')
            assert z.shape[1]==4096 and l.shape[1]==4096;assert saved['raw_paths'].shape==z.shape and saved['logits'].shape==l.shape
            # Same inference GPU and chunking as the formal evaluator.
            np.testing.assert_array_equal(z,saved['raw_paths']);np.testing.assert_array_equal(l,saved['logits'])
            members=decode(z,l,.5,cfg,members_only=True);data=np.zeros((3,4096,3,6),np.float32);k=np.zeros((3,4096),np.int8)
            for i,member in enumerate(members):
                for q,row in enumerate(member):k[i,q]=len(row);data[i,q,:len(row)]=row
            np.savez_compressed(EXP/f'cache/full_members_{pair}_{branch}.npz',paths=data,n_paths=k,old_medoid=saved['medoid'],old_prediction=saved['prediction'],old_predicted_k=saved['predicted_k'])
            evidence.append(dict(pair=pair,branch=branch,queries=4096,members=3,raw_slots=z.shape[2],raw_paths_bitwise_equal=True,logits_bitwise_equal=True,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()));print('REGENERATED_ALL_MEMBERS',pair,branch,flush=True)
    (EXP/'results/member_regeneration.json').write_text(json.dumps(dict(passed=True,epoch=time.time(),results=evidence,wall_s=time.perf_counter()-start,training_steps=0,new_measurements=0,full_queries=True),indent=2)+'\n')

if __name__=='__main__':run()
