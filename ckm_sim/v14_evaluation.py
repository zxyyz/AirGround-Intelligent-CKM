"""Evaluation-only private references, float64 metrics and subset decomposition."""
import numpy as np
from .config import array_hash
from .metrics import squared_gospa,parameter_rmse,path_squared_distance

class PrivateEvaluator:
    def __init__(self,path,cfg):
        with np.load(path) as d:
            self.queries=d['queries'];self.weights=d['weights'];z=d['reference'];o=d['offsets']
            self.__reference=[z[a:b].astype(np.float64) for a,b in zip(o[:-1],o[1:])]
        self.cfg=cfg;self.query_hash=array_hash(self.queries,self.weights)

    def evaluate(self,predictions):
        predictions=[np.asarray(p,np.float64) for p in predictions]
        reference=self.__reference;weights=self.weights;cfg=self.cfg
        result,errors=parameter_rmse(predictions,reference,weights,cfg)
        kt=np.array([len(p) for p in reference]);kp=np.array([len(p) for p in predictions])
        detail=[];confusion=np.zeros((4,4),int);cutoff=cfg['metric']['gospa_c']**2
        for p,t,a,b in zip(predictions,reference,kp,kt):
            e,pairs,miss,false=squared_gospa(p,t,cfg);matched=sum(float(path_squared_distance(p[i],t[j],cfg)) for i,j in pairs)
            card=cutoff/2*abs(a-b);extra=cutoff*(min(a,b)-len(pairs));unmatched=cutoff/2*(miss+false)
            if not np.isclose(e,matched+unmatched,rtol=1e-10,atol=1e-10):raise ValueError('Squared GOSPA decomposition failed')
            if not np.isclose(unmatched,card+extra,rtol=0,atol=1e-10):raise ValueError('Unmatched decomposition failed')
            detail.append([e,matched,miss,false,card,extra,len(pairs)])
            confusion[b,a]+=1
        detail=np.array(detail);subsets=[]
        for name,mask in [('all',kt>=0),('nonempty',kt>0),('K1',kt==1),('K2plus',kt>=2),('empty',kt==0)]:
            ix=np.flatnonzero(mask)
            if not len(ix):continue
            w=weights[ix]/weights[ix].sum();r,_=parameter_rmse([predictions[i] for i in ix],[reference[i] for i in ix],w,cfg)
            subsets.append({'subset':name,'n_queries':len(ix),'population_mass':float(weights[ix].sum()),**r,
                'matched_parameter_cost':float(w@detail[ix,1]),'miss_cost':float(cutoff/2*(w@detail[ix,2])),
                'false_cost':float(cutoff/2*(w@detail[ix,3])),'cardinality_cost':float(w@detail[ix,4]),
                'additional_unmatched_cost':float(w@detail[ix,5]),'count_accuracy':float(w@(kp[ix]==kt[ix]))})
        return result,subsets,{'squared_gospa':errors,'true_k':kt,'predicted_k':kp,'decomposition':detail,'confusion':confusion,
            'prediction_lengths':kp,'prediction_values':np.concatenate(predictions) if kp.sum() else np.empty((0,6))}

def build_private_queries(path,cfg,g,world,spatial_holdout=False):
    from .propagation import Evaluator
    from .v14_protocol import heldout_blocks
    q,w=g.sample_safe_queries(cfg['queries']['count']*(5 if spatial_holdout else 1),cfg['queries']['seed'])
    if spatial_holdout:
        q=q[heldout_blocks(q)][:cfg['queries']['count']]
        if len(q)!=cfg['queries']['count']:raise ValueError('Not enough spatial holdout queries')
        w=np.full(len(q),1/len(q))
    evaluator=Evaluator(cfg,world,q,w);evaluator.save_private(path)
    return PrivateEvaluator(path,cfg)
