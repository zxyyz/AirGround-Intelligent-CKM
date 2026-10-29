from pathlib import Path
import copy,csv,hashlib,json,os,sys,time,traceback
EXP=Path(__file__).absolute().parents[1]
ROOT=EXP
sys.path.insert(0,str(ROOT))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda a:a.tolist())+'\n');tmp.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def config(scene='H0'):
    assert scene=='H0'
    return json.loads((EXP/'configs/effective_config.json').read_text())
def csvwrite(path,rows):
    rows=list(rows);assert rows
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def check_deadline():
    if (EXP/'STOP').exists():raise RuntimeError('Explicit local stop requested')
def stage(name):
    check_deadline();s=json.loads((EXP/'STATUS.json').read_text());s.update(stage=name,last_progress_epoch=time.time());save(EXP/'STATUS.json',s)
def failure(name,exc):
    with (EXP/'failures.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=time.time(),stage=name,error=repr(exc),traceback=traceback.format_exc()))+'\n')
