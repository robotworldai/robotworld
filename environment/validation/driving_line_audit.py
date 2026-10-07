"""Independent polygon-clipping check of declared footprint/paint overlap.
Unlike the online SAT test, this computes intersection area from raw poses.
It validates the declared safety rectangle, not visible mesh or tire contact.
"""
import argparse,json,math
from pathlib import Path

def clip(poly,axis,bound,keep_greater):
 out=[]
 if not poly:return out
 prev=poly[-1];inside=lambda p:p[axis]>=bound if keep_greater else p[axis]<=bound
 for cur in poly:
  a,b=inside(prev),inside(cur)
  if a!=b:
   t=(bound-prev[axis])/(cur[axis]-prev[axis]);out.append([prev[i]+t*(cur[i]-prev[i]) for i in range(2)])
  if b:out.append(cur)
  prev=cur
 return out

def intersection_area(poly,bounds):
 xmin,xmax,ymin,ymax=bounds
 for axis,bound,greater in [(0,xmin,True),(0,xmax,False),(1,ymin,True),(1,ymax,False)]:poly=clip(poly,axis,bound,greater)
 if len(poly)<3:return 0.
 return abs(sum(p[0]*q[1]-q[0]*p[1] for p,q in zip(poly,poly[1:]+poly[:1])))/2

def bridge_violations(spec,state):
 # Independently compare each tread interval with the unpainted support interval.
 # This is a wheel-link geometric audit, not a measured contact-force claim.
 b=spec.get('bridge')
 if b is None:return []
 wheels=state.get('wheel_positions')
 if wheels is None or len(wheels)!=4:raise ValueError('Four wheel links required for bridge audit')
 issues=[]
 for name,w in wheels.items():
  if not b['x'][0]+.015<w[0]<b['x'][1]-.015:continue
  center=b['track_y'][1 if 'left' in name else 0]
  lo=center-b['beam_width']/2+.012;hi=center+b['beam_width']/2-.012
  if w[1]-b['tire_half_width']<=lo or w[1]+b['tire_half_width']>=hi:
   issues.append('tire_touched_rail_edge:'+name)
  if w[2]<b['height']+.015:issues.append('wheel_dropped_from_beam:'+name)
 return issues

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();spec=json.loads((a.run/'scenario.json').read_text())
 hx,hy=spec['footprint'];width=spec['line_width'];rects=[]
 for first,last in spec.get('forbidden_lines',[]):
  dx=abs(first[0]-last[0]);dy=abs(first[1]-last[1]);assert dx==0 or dy==0,'This audit requires axis-aligned paint'
  x=(first[0]+last[0])/2;y=(first[1]+last[1])/2;sx=max(dx,width)/2;sy=max(dy,width)/2
  rects.append([x-sx,x+sx,y-sy,y+sy])
 first_overlap=None;first_bridge_violation=None;checked=0
 for line in (a.run/'events/scoring.jsonl').read_text().splitlines():
  event=json.loads(line)
  if event['kind']!='scoring_step':continue
  state=event['payload']['state'];x,y=state['position'][:2];c=math.cos(state['yaw']);s=math.sin(state['yaw'])
  poly=[[x+u*c-v*s,y+u*s+v*c] for u,v in [(-hx,-hy),(hx,-hy),(hx,hy),(-hx,hy)]];checked+=1
  issues=bridge_violations(spec,state)
  if issues and first_bridge_violation is None:first_bridge_violation={'step':state['step'],'issues':issues}
  areas=[intersection_area(poly,r) for r in rects]
  if first_overlap is None and max(areas,default=0.)>1e-12:
   first_overlap={'step':state['step'],'paint_indices':[i for i,v in enumerate(areas) if v>1e-12],'intersection_areas_m2':areas,'footprint_polygon':poly}
 saved=json.loads((a.run/'result.json').read_text())['robotworld'];online=saved.get('line_touched');matches=(online is None and first_overlap is None) or bool(online and first_overlap and int(online.split('_')[-1]) in first_overlap['paint_indices'] and saved['step']==first_overlap['step'])
 bridge_matches=True
 if 'bridge' in spec:
  reason=saved.get('failure_reason') or ''
  if first_bridge_violation:
   bridge_matches=saved['step']==first_bridge_violation['step'] and (reason in first_bridge_violation['issues'] or reason=='fell_from_bridge')
  else:bridge_matches=not reason.startswith(('tire_touched_rail_edge:','wheel_dropped_from_beam:'))
  matches=matches and bridge_matches
 result={'first_bridge_violation':first_bridge_violation,'bridge_tread_interval_matches':bridge_matches if 'bridge' in spec else None,'algorithm':'Sutherland-Hodgman clipping + shoelace area for paint; independent tread/support interval comparisons for bridge','scope':'declared 2D safety footprint plus bridge tread/support intervals; not measured physical contact; zero-area paint tangency is not certified','checked_control_steps':checked,'footprint_half_extents_m':[hx,hy],'line_width_m':width,'first_positive_area_overlap':first_overlap,'online_line_touched':online,'online_terminal_step':saved['step'],'matches':matches}
 (a.run/'independent-line-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));raise SystemExit(0 if matches else 1)
if __name__=='__main__':main()
