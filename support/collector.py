from common import *
import numpy as np
from physics import frontend,fold

class Receiver:
    def __init__(self,cfg):self.estimator,self.cal=frontend(cfg);self.batch_size=cfg["runtime"].get("receiver_batch_size",8)
    def estimate(self,signals,pilots):
        zz=[];qq=[]
        for i in range(0,len(signals),self.batch_size):
            try:z,q=self.estimator.estimate_many(signals[i:i+self.batch_size],pilots)
            except Exception as exc:
                print('RF_BATCH_FALLBACK',repr(exc),flush=True);z=[];q=[]
                for s in signals[i:i+self.batch_size]:
                    try:a,b=self.estimator.estimate(s,pilots)
                    except Exception as err:a=np.empty((0,6));b=dict(valid=False,reason=repr(err))
                    z.append(a);q.append(b)
            zz.extend(fold(a) for a in z);qq.extend(q)
        return zz,qq
