"""Execute untouched pure upstream predicate functions on numeric/Boolean fixtures.
AST extraction skips simulator imports, not function statements. Physics contacts
and robot world transforms remain a separate runtime validation requirement.
"""
import argparse,ast,hashlib,itertools,json,math
from pathlib import Path
from types import SimpleNamespace as NS
from functools import reduce
import numpy as np
import torch
WORLD=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
 root=WORLD/'third_party/benchmarks/robolab/checkout/robolab';ns={'torch':torch,'np':np,'DEBUG':False,'reduce':reduce};sources=[]
 for file,names in [(root/'core/task/predicate_logic.py',['evaluate_logicals','evaluate_logicals_vectorized','center_of','stationary','upright','in_contact','gripper_detached','_and','_not','check_stacked']),(root/'core/utils/geometry_utils.py',['spatial_condition_check_vector_based']),(WORLD/'third_party/dependencies/isaaclab22/checkout/source/isaaclab/isaaclab/utils/math.py',['unmake_pose'])]:
  source=file.read_text();tree=ast.parse(source);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),'exec'),ns)
  sources.append({'file':str(file.relative_to(WORLD)),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'functions':[n.name for n in nodes]})
 rows=[]
 def check(task,label,actual,expected):rows.append({'task':task,'fixture':label,'actual':bool(actual),'expected':expected,'passed':bool(actual)==expected})
 for task,n,k,logical in [('FruitsOnPlate3Task',7,3,'choose'),('PutTwoMugsOnShelfTask',3,2,'choose'),('ToolOrganizationTask',2,2,'all'),('NonHammerToolsInRightBinTask',2,2,'all'),('FoodPacking2CansTask',2,2,'all'),('ClutterPlasticTask',3,3,'all')]:
  for bits in itertools.product([False,True],repeat=n):
   expected=sum(bits)==k
   check(task,str(bits)+' scalar',ns['evaluate_logicals'](list(bits),logical,k),expected)
   check(task,str(bits)+' batched',ns['evaluate_logicals_vectorized']([torch.tensor([v]) for v in bits],logical,k)[0],expected)
 for x,y,expected in [(0,1,True),(0,-1,False),(1,0,False),(0,0,False),(.5,1,True),(2,1,False)]:
  a1=torch.eye(4);a1[:3,3]=torch.tensor([x,y,0]);a2=torch.eye(4)
  for batch in [False,True]:check('RubiksCubeLeftOfBowlTask',f'robot_relative_xy={x,y};batch={batch}',ns['spatial_condition_check_vector_based'](a1[None] if batch else a1,a2[None] if batch else a2,'left_of'),expected)
 for angle,expected in [(0.,True),(.099,True),(.101,False),(math.pi/2,False),(math.pi,False)]:
  for eid in [0,None]:
   quat=torch.tensor([math.cos(angle/2),math.sin(angle/2),0,0]);world=NS(get_pose=lambda obj,env_id:(torch.zeros(3),quat if env_id is not None else quat[None]))
   check('ReorientWhiteMugsTask',f'tilt={angle};env_id={eid}',ns['upright'](world,'mug',env_id=eid),expected)
 for x,y,expected in [(0,0,True),(.049,0,True),(.051,0,False),(0,-.051,False),(.049,.049,True)]:
  for eid in [0,None]:
   world=NS(get_centroid=lambda obj,env_id:torch.tensor([x,y,0]) if obj=='mug' and env_id is not None else torch.tensor([[x,y,0]]) if obj=='mug' else torch.zeros(3) if env_id is not None else torch.zeros(1,3))
   check('WhiteMugInCenterOfTableTask',f'xy_error={x,y};env_id={eid}',ns['center_of'](world,'mug','table',tolerance=.05,env_id=eid),expected)
 for left,right in itertools.product([False,True],repeat=2):
  for eid in [0,None]:
   def contact(a,b,threshold,env_id):v=left if b=='left' else right;return v if env_id is not None else torch.tensor([v])
   world=NS(in_contact=contact)
   check('shared_gripper_detached',f'contacts={left,right};env_id={eid}',ns['gripper_detached'](world,'object',['left','right'],env_id=eid),not left and not right)
 # Original stacking loop; geometric leaves carry explicit synthetic heights/contact.
 colors=['red','blue','green','yellow']
 ns['above_top']=lambda world,a,b,*args,**kw:world.z[a]>world.z[b]
 ns['above_bottom']=ns['above_top']
 for permutation in itertools.permutations(colors):
  z={c:i for i,c in enumerate(permutation)}
  world=NS(z=z,in_contact=lambda a,b,env_id:abs(z[a]-z[b])==1)
  check('BlockStackingSpecifiedOrderTask','bottom_to_top='+','.join(permutation),ns['check_stacked'](world,colors,order='bottom_to_top',env_id=0),list(permutation)==colors)
 report={'kind':'unmodified extracted pure functions; synthetic numeric/Boolean states; not a physics proof','sources':sources,'tests':rows,'total':len(rows),'passed':sum(r['passed'] for r in rows)}
 a.output.write_text(json.dumps(report,indent=2));print(json.dumps({'total':len(rows),'passed':report['passed']}));raise SystemExit(0 if report['passed']==len(rows) else 1)
if __name__=='__main__':main()
