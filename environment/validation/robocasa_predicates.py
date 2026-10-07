"""Contract tests of unmodified upstream _check_success (mocked geometric leaves).
This proves boolean composition/boundaries only, not geometric/physics correctness.
Run in the pinned RoboCasa image; never imported by an evaluation episode.
"""
import argparse,inspect,json,math
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import robocasa

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 names=['CountertopCleanup','SortingCleanup','CoffeeSetupMug','CloseDrawer','NavigateKitchen','PackIdenticalLunches','OrganizeMugsByHandle','LoadDishwasher','MicrowaveCorrectMeal','ResetCabinetDoors']
 rows=[];sources=[]
 for name in names:
  cls=getattr(robocasa,name);fn=cls._check_success;g=fn.__globals__;calls=set();override={}
  def leaf(key,default=True):calls.add(key);return override.get(key,default)
  def fixture(label):
   return NS(rot=0.,is_closed=lambda *args,**kw:leaf(label+':closed'),check_rack_contact=lambda env,obj,**kw:leaf(label+':rack:'+obj),check_receptacle_placement_for_pouring=lambda env,obj:leaf(label+':placement:'+obj),get_state=lambda:{'turned_on':leaf(label+':on')},get_door_state=lambda **kw:{'joint0':override.get('drawer_position',0.0)})
  e=NS(**{k:fixture(k) for k in ['drawer','cab','cab2','cab3','cabinet','sink','fridge','coffee_machine','dishwasher','microwave']})
  e.behavior='close' if name=='CloseDrawer' else 'counter_to_machine';e.target_bowl=0
  e.target_pos=np.zeros(3);e.target_ori=np.zeros(3);e.obj_body_id={'mug_counter1':0}
  e.sim=NS(model=NS(body_name2id=lambda key:0),data=NS(body_xpos=np.zeros((1,3)),body_xmat=np.eye(3).reshape(1,9),body_xquat=np.array([[math.cos(math.pi/4),0,0,math.sin(math.pi/4)]])))
  packed={(f'{kind}{i}',f'tupperware{i}') for kind in ['vegetable','meat'] for i in [0,1]}
  class Objects:
   def obj_inside_of(self,env,obj,fixture):return leaf('inside:'+obj)
   def check_obj_any_counter_contact(self,env,obj):return leaf('counter_contact:'+obj,False)
   def check_obj_in_receptacle(self,env,obj,rec):return leaf('in:'+obj+':'+rec,(obj,rec) in packed if name=='PackIdenticalLunches' else True)
   def gripper_obj_far(self,env,obj_name='obj'):return leaf('released:'+obj_name)
  def check(label,expected):
   actual=bool(fn(e));rows.append({'task':name,'fixture':label,'expected':expected,'actual':actual,'passed':actual==expected})
  with patch.dict(g,{'OU':Objects()}):
   check('all_native_requirements_met',True)
   base_calls=set(calls)
   if name not in ['NavigateKitchen','OrganizeMugsByHandle','CloseDrawer','PackIdenticalLunches']:
    for key in sorted(base_calls):
     override[key]=key.startswith('counter_contact:');check('missing_requirement:'+key,False);override.clear()
   if name=='CloseDrawer':
    for pos,expected in [(0.,True),(.05,True),(.050001,False),(.8,False)]:override['drawer_position']=pos;check('normalized_opening='+str(pos),expected)
   if name=='NavigateKitchen':
    for x,yaw,expected in [(0.2,0,True),(.200001,0,False),(0,math.acos(.98)-1e-6,True),(0,math.acos(.98)+1e-6,False),(0,math.pi,False)]:
     e.sim.data.body_xpos[0,0]=x;e.target_ori[2]=yaw;check(f'distance={x};yaw_error={yaw}',expected)
   if name=='OrganizeMugsByHandle':
    for cab_angle in [0,300]:
     e.cabinet.rot=math.radians(cab_angle)
     for rel,expected in [(49.9,False),(50.1,True),(90,True),(129.9,True),(130.1,False),(270,False)]:
      yaw=math.radians(cab_angle+rel);e.sim.data.body_xquat[0]=[math.cos(yaw/2),0,0,math.sin(yaw/2)];check(f'cabinet={cab_angle};handle_relative={rel}',expected)
    yaw=math.radians(390);e.sim.data.body_xquat[0]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
    for key in ['inside:mug_counter1','released:mug_counter1']:override[key]=False;check('missing_requirement:'+key,False);override.clear()
   if name=='PackIdenticalLunches':
    for key in sorted(base_calls):
     if key.startswith('released:'):override[key]=False;expected=False
     else:
      _,obj,rec=key.split(':');override[key]=(obj,rec) not in packed;expected=False
     check('removed_or_extra_assignment:'+key,expected);override.clear()
    override.update({'in:vegetable1:tupperware1':False,'in:vegetable0:tupperware1':True});check('same_vegetable_counted_in_two_boxes',False);override.clear()
   if name=='MicrowaveCorrectMeal':
    e.target_bowl=1;check('target_bowl_1_met',True);override['inside:bowl1']=False;check('wrong_bowl_0_inserted',False)
  sources.append({'task':name,'bound_method':fn.__qualname__,'module':fn.__module__,'source_file':inspect.getsourcefile(fn),'source_line':inspect.getsourcelines(fn)[1],'predicate':inspect.getsource(fn),'language_method':inspect.getsource(cls.get_ep_meta)})
 result={'test_kind':'actual native method; mocked geometric leaves; not a simulator-state or physics proof','tests':rows,'total':len(rows),'passed':sum(r['passed'] for r in rows)}
 (a.output/'robocasa-predicate-contracts.json').write_text(json.dumps(result,indent=2));(a.output/'robocasa-bound-methods.json').write_text(json.dumps(sources,indent=2))
 print(json.dumps({'tests':len(rows),'passed':result['passed']}));raise SystemExit(0 if all(r['passed'] for r in rows) else 1)
if __name__=='__main__':main()
