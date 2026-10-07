"""Executed by isolated RTAMT Python; imports upstream monitor unchanged."""
import importlib.util, json, math, sys
from pathlib import Path
import rtamt
# RTAMT 0.3.5 renamed the same public factory; leave upstream monitor untouched.
rtamt.STLDenseTimeSpecification=rtamt.StlDenseTimeSpecification
p=json.load(sys.stdin)
file=Path(p['source'])/'Evaluation/eval_monitor/stl_dense_offline.py'
spec=importlib.util.spec_from_file_location('upstream_monitor',file)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
# Upstream uses `is` on task strings, so explicitly use the matching interned literal.
name=sys.intern(p['task']);monitor=m.stl_dense_offline_monitor(task_name=name)
sequence=monitor.compute_robustness(p['trace'])
values=[float(row[1]) for row in sequence]
rob=(max if name=='FrankaPegInHole' else min)(values)
# Faithfully preserve optimizer.py's strict >0, and its special max for PegInHole.
json.dump({'success':rob>0,'robustness':rob if math.isfinite(rob) else str(rob),'aggregation':'max' if name=='FrankaPegInHole' else 'min','status':'scored','robustness_sequence':[[float(t),float(v) if math.isfinite(v) else str(v)] for t,v in sequence]},sys.stdout,allow_nan=False)
