from pathlib import Path
import copy,csv,hashlib,json,os,sys,time,traceback
import numpy as np
EXP=Path(__file__).resolve().parents[1]
ROOT=EXP.parents[1]
OLD=ROOT/'experiments/paper_formal_240_20261017'
sys.path.insert(0,str(ROOT))
BRANCHES=['I-D','I-G','C-G','P-G']
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda a:a.tolist())+'\n');tmp.replace(path)
def load(path):return json.loads(Path(path).read_text())
def csvwrite(path,rows):
    rows=list(rows);keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def cfg():return load(OLD/'configs/effective_config.json')
def stage(name,**kw):
    p=EXP/'STATUS.json';s=load(p);s.update(stage=name,last_progress_epoch=time.time(),**kw);save(p,s)
def failure(where,error):
    with (EXP/'logs/failures.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=time.time(),stage=where,error=repr(error),traceback=traceback.format_exc()))+'\n')
def scales():
    p=EXP/'configs/scales_resolved.json'
    return load(p)['scales'] if p.exists() else [a for a in load(EXP/'configs/protocol.json')['scales'] if a['id']=='S0']
