"""Frozen architectures; identical replay sampling on identical I data."""
from common import *
import numpy as np
from ckm_sim.geometry import GeometryMap
from public_multibounce_geometry import PublicGeometry
from learning import Committee
from models import ExperimentCommittee

def initialization_seed(pair,member,variant):
    if variant=='M3':return pair*100+810+member
    assert variant=='D'
    return int.from_bytes(hashlib.sha256(f'D-replay:init:{pair}:{member}'.encode()).digest()[:8],'big')&((1<<63)-1)

def make_model(cfg,pair,variant):
    if variant=='D':
        c=copy.deepcopy(cfg);c['model']['variant']='direct_set'
        m=Committee(c,pair,'Lreplay');m.variant='D'
        from ckm_sim.model import SetCKM
        import torch
        # Keep the paired replay generators; replace initialization only.
        m.models=[];m.optimizers=[]
        for i in range(c['model']['committee_size']):
            torch.manual_seed(initialization_seed(pair,i,'D'))
            model=SetCKM(c).to(m.device);m.models.append(model)
            m.optimizers.append(torch.optim.AdamW(model.parameters(),lr=c['training']['learning_rate'],weight_decay=c['training']['weight_decay']))
        assert all(type(a) is SetCKM for a in m.models)
        return m
    assert variant=='M3'
    return ExperimentCommittee(cfg,pair,'M3')

class Features:
    def __init__(self,cfg,variant):
        self.base=GeometryMap(cfg);self.variant=variant;self.public=PublicGeometry(cfg,self.base)
        self.cache={};self.costs=[]
    def __getattr__(self,name):return getattr(self.base,name)
    def features(self,x):
        x=np.atleast_2d(x)
        missing={tuple(a):a for a in x if tuple(a) not in self.cache}
        if missing:
            keys=list(missing);xx=np.array([missing[k] for k in keys])
            if self.variant=='D':f=self.base.features(xx)
            else:
                f,stats=self.public.features(xx,2,64);self.costs.extend(stats)
            for key,value in zip(keys,f):self.cache[key]=value
        return np.asarray([self.cache[tuple(a)] for a in x],np.float32)
