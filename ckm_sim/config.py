"""Inheritance, provenance and scientific freezing."""
import copy
import hashlib
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()

def merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out

def load_config(path, seen=()):
    path = Path(path).resolve()
    if path in seen:
        raise ValueError('Cyclic configuration')
    cfg = yaml.safe_load(path.read_text())
    parent = cfg.pop('inherits', None)
    cfg = merge(load_config(path.parent / parent, (*seen, path)), cfg) if parent else cfg
    if 'frozen_sha256' in cfg:
        expected = cfg.pop('frozen_sha256')
        if canonical_hash(cfg) != expected:
            raise ValueError('Frozen configuration was modified')
        cfg['frozen_sha256'] = expected
    return cfg

def scientific_hash(cfg):
    return canonical_hash({k:v for k,v in cfg.items() if k not in ['runtime','frozen_sha256']})

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=lambda x:x.tolist())+'\n')

def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(str((a.shape, str(a.dtype))).encode()); h.update(a.tobytes())
    return h.hexdigest()
