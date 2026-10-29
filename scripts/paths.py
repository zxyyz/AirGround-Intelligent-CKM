"""Portable experiment paths and deterministic utilities."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,os,sys
EXP=Path(__file__).resolve().parents[1]
ROOT=EXP
sys.path.insert(0,str(EXP))
sys.path.append(str(EXP/'support'))
VARIANTS=['Coord','RawMap','GeoPool','GeoDirect','Full']
THRESHOLDS=[.10,.25,.35,.50,.65,.75,.90]
LRS=[.0003,.001,.003]
def config(): return json.loads((EXP/'configs/effective_config.json').read_text())
def stamp(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.tolist())+'\n');tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def seed(*parts):return int.from_bytes(hashlib.sha256(':'.join(map(str,('geometry-attribution-20261027',*parts))).encode()).digest()[:8],'big')%(2**63-1)
def run_dir(stage,variant,lr,pair,ratio=1.):return EXP/f'runs/{stage}/ratio_{ratio:.2f}/{variant}/lr_{lr:g}/{pair}'
