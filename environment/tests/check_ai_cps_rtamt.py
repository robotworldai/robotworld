"""Container-only regression: native monitor, strict >0 and task aggregation."""
import json, subprocess, sys
from pathlib import Path
world=Path(__file__).resolve().parents[2]
worker=world/'environment/benchmarks/ai_cps/score_worker.py'
source=world/'third_party/benchmarks/ai_cps/checkout'
results=[]
for name,limit,aggregate in [('FrankaBallCatching',.1,'min'),('FrankaBallBalancing',.25,'min'),('FrankaPegInHole',.1,'max')]:
    for distance in (0.,limit,limit+.1):
        payload={'task':name,'source':str(source),'trace':[[i+1,distance] for i in range(299)]}
        p=subprocess.run(['/opt/rtamt/bin/python',str(worker)],input=json.dumps(payload),capture_output=True,text=True,check=True)
        r=json.loads(p.stdout)
        assert r['success']==(distance<limit),(name,distance,r)
        assert abs(r['robustness']-(limit-distance))<1e-8
        assert r['aggregation']==aggregate
        results.append({'task':name,'distance':distance,'success':r['success'],'robustness':r['robustness']})
print(json.dumps({'passed':len(results),'cases':results},indent=2))
