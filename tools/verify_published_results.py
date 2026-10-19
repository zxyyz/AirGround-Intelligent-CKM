"""Verify aggregate results without raw measurement data."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean,stdev
R=Path(__file__).resolve().parents[1]
base=R/"experiments/paper_formal_240_20261017/results"
rows=list(csv.DictReader((base/"endpoint.csv").open()))
d=defaultdict(dict)
for row in rows:
    d[int(row["pair"])][row["branch"]]=float(row["global_gospa"])
assert len(d)==10
for seed,values in d.items():
    assert all(branch in values for branch in ("I-D","I-G","C-G")),seed
for branch,label in (("I-D","Baseline"),("I-G","Geometry"),("C-G","Cooperative")):
    values=[d[s][branch] for s in sorted(d)]
    print(f"{label:11s}: global GOSPA = {mean(values):.4f} +/- {stdev(values):.4f}")
for method,control,label in (("I-G","I-D","Geometry vs Baseline"),("C-G","I-G","Cooperative vs Geometry")):
    vals=[100*(1-d[s][method]/d[s][control]) for s in sorted(d)]
    print(f"{label:24s}: improvement = {mean(vals):.4f}%; positive seeds {sum(x>0 for x in vals)}/10")
assert abs(mean([100*(1-d[s]["I-G"]/d[s]["I-D"]) for s in d])-52.87211540353517)<1e-9
assert abs(mean([100*(1-d[s]["C-G"]/d[s]["I-G"]) for s in d])-9.066053208458385)<1e-9
print("CHECK PASSED")
