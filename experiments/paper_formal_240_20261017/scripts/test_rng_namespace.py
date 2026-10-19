"""Verify independent initialization and compatibility before refreezing."""
from common import *
import importlib.util
import torch
from formal_models import initialization_seed,make_model

def run():
    torch.set_num_threads(2);cfg=config();rows=[];seen={}
    for pair in [5099,*range(5101,5111)]:
        for i in range(3):
            row=dict(pair=pair,member=i,D_initialization=initialization_seed(pair,i,'D'),M3_initialization=initialization_seed(pair,i,'M3'),paired_sampler=pair*100+30+i)
            rows.append(row)
            for key in ['D_initialization','M3_initialization','paired_sampler']:
                value=row[key];assert value not in seen,(value,seen.get(value),row)
                seen[value]=(pair,i,key)
    save(EXP/'configs/initialization_seed_map.json',dict(rows=rows,algorithm='SHA256 D-replay:init:pair:member, first 8 bytes big endian masked to 63 bits',M3_and_paired_sampler_unchanged=True))
    spec=importlib.util.spec_from_file_location('original_formal_models',EXP/'archives/initialization_attempt_v1/scripts_v1/formal_models.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    a=old.make_model(cfg,5099,'M3');b=make_model(cfg,5099,'M3')
    for x,y in zip(a.models,b.models):
        for key,value in x.state_dict().items():torch.testing.assert_close(value,y.state_dict()[key],rtol=0,atol=0)
    for x,y in zip(a.samplers,b.samplers):assert torch.equal(x.get_state(),y.get_state())
    d=make_model(cfg,5099,'D');assert [r.initial_seed() for r in d.samplers]==[5099*100+30+i for i in range(3)]
    from ckm_sim.model import SetCKM
    for i,model in enumerate(d.models):
        torch.manual_seed(initialization_seed(5099,i,'D'));expected=SetCKM(d.cfg).to(d.device)
        for key,value in expected.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
    save(EXP/'results/rng_namespace_audit.json',dict(passed=True,epoch=time.time(),unique_initialization_and_sampler_streams=len(seen),pairs=11,formal_pairs=10,M3_initial_weights_bitwise_unchanged=True,M3_samplers_unchanged=True,D_actual_initial_weights_verified=True,paired_replay_samplers_unchanged=True,seed_map_sha256=sha(EXP/'configs/initialization_seed_map.json')))
    print('RNG_NAMESPACE_AND_COMPATIBILITY_PASS',flush=True)

if __name__=='__main__':run()
