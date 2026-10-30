from pathlib import Path
import sys,json
EXT=Path(__file__).resolve().parents[1]
BASE=EXT.parents[1]
sys.path.insert(0,str(BASE/'scripts'))
from paths import sha,save,stamp

def directory(variant,lr,pair,ratio):
    return EXT/f'runs/formal/ratio_{ratio:.2f}/{variant}/lr_{lr:g}/{pair}'
def read(path):return json.loads(Path(path).read_text())
def tasks():return read(EXT/'manifests/tasks.json')
def assert_freeze():
    f=read(BASE/'configs/freeze.json')
    for name,h in {**f['training_source_hashes'],**f['asset_hashes'],**f['batch_hashes']}.items():assert sha(BASE/name)==h,name
    assert f['steps']==56800 and f['learning_rates']['GeoPool']==.003 and f['learning_rates']['GeoDirect']==.0003
    assert f['thresholds']['GeoPool']==.9 and f['thresholds']['GeoDirect']==.5
    return f
