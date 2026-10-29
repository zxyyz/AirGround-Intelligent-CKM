"""Float64 audit oracles, not training replacements. Inputs are finite arrays."""
import itertools
import numpy as np
from scipy.special import logsumexp


def injection(cost, top_three=True):
    """Exact K<=3 assignment, returned as one slot index per target."""
    cost=np.asarray(cost,dtype=float)
    n,k=cost.shape
    if not (0<=k<=3 and n>=k): raise ValueError('Require K<=3 and at least K slots')
    if not np.isfinite(cost).all(): raise ValueError('Nonfinite assignment cost')
    if k==0: return 0., ()
    choices=[np.argsort(cost[:,j],kind='stable')[:min(3,n)] if top_three else range(n) for j in range(k)]
    feasible=(p for p in itertools.product(*choices) if len(set(p))==k)
    best=min(feasible,key=lambda p:sum(cost[i,j] for j,i in enumerate(p)))
    return float(sum(cost[i,j] for j,i in enumerate(best))),tuple(map(int,best))


def reference_loss(distance, logits, lp=1.):
    """Untruncated full injection + sum BCE, exactly the legacy objective."""
    logits=np.asarray(logits,dtype=float)
    best,assignment=injection(np.asarray(distance)-lp*logits[:,None])
    return float(lp*np.logaddexp(0,logits).sum()+best),assignment


def candidate_loss(distance, logits, weights=None, cutoff=3., lp=1.,
                   miss_penalty=4.5, coverage_weight=.25, tail=.05,
                   coverage_temperature=1., positive_weight=1.):
    """Unfrozen L_new proposal: partial assignment + smooth recall coverage.

    Pass active, trainable slots only. Coverage is intentionally many-to-one;
    the primary assignment remains injective. Never use this as GOSPA.
    """
    d=np.asarray(distance,dtype=float); logits=np.asarray(logits,dtype=float)
    n,k=d.shape
    if n<1 or not 0<=k<=3 or len(logits)!=n: raise ValueError('Invalid active set')
    if not np.isfinite(d).all() or not np.isfinite(logits).all() or np.any(d<0):
        raise ValueError('Expected finite nonnegative squared distances')
    w=np.ones(k) if weights is None else np.asarray(weights,dtype=float)
    if w.shape!=(k,) or np.any(w<=0) or not np.isfinite(w).all(): raise ValueError('Invalid weights')
    if min(lp,miss_penalty,coverage_weight,tail,coverage_temperature,positive_weight)<=0:
        raise ValueError('All candidate constants must be positive')
    rho=np.minimum(d,cutoff**2)+tail*np.log1p(d)
    base=lp*np.logaddexp(0,logits).sum()+miss_penalty*w.sum()
    best=0.; selected=()
    for size in range(1,min(n,k)+1):
        for targets in itertools.combinations(range(k),size):
            t=np.array(targets)
            value,slots=injection(rho[:,t]*w[t]-lp*logits[:,None]-miss_penalty*w[t])
            if value<best:
                best=value; selected=tuple(zip(slots,targets))
    if k:
        costs=rho+positive_weight*np.logaddexp(0,-logits)[:,None]
        cover=-coverage_temperature*(logsumexp(-costs/coverage_temperature,axis=0)-np.log(n))
        recall=float(w@cover)
    else: recall=0.
    return float(base+best+coverage_weight*recall),selected
