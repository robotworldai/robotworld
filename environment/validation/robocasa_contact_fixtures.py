"""Paused geometric/contact predicate fixtures, not demonstrations of a controller.
Contact placement uses real MuJoCo collision queries and records contact depths.
It does not prove dynamic stability, reachability, or a correct action sequence.
"""
import argparse,json
from pathlib import Path
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import robocasa
from robocasa.utils import object_utils as OU

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--tasks',default='PackIdenticalLunches,CountertopCleanup,MicrowaveCorrectMeal,LoadDishwasher');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=[];errors=[]
 for task in a.tasks.split(','):
  wrapper=writer=None
  try:
   wrapper=gym.make('robocasa/'+task,split='pretrain',seed=7,horizon=600);wrapper.reset(seed=7);e=wrapper.unwrapped.env;base=e.sim.get_state()
   def pos(name):return e.sim.data.body_xpos[e.obj_body_id[name]].copy()
   def put(name,xyz):
    joint=e.objects[name].joints[0];q=e.sim.data.get_joint_qpos(joint).copy();q[:3]=xyz;e.sim.data.set_joint_qpos(joint,q);e.sim.forward()
   def center(f):
    p0,px,py,pz=next(iter(f.get_int_sites(relative=False).values()));return p0+((px-p0)+(py-p0)+(pz-p0))/2
   def contact(name,recep,side):
    origin=pos(recep);radius=e.objects[recep].horizontal_radius
    for dz in np.linspace(.25,-.10,351):
     put(name,origin+[side*radius*.28,0,dz])
     if OU.check_obj_in_receptacle(e,name,recep):return
    raise RuntimeError('Cannot construct native contact fixture '+name+' in '+recep)
   if task=='PackIdenticalLunches':
    for i in range(2):
     contact('vegetable'+str(i),'tupperware'+str(i),-1);contact('meat'+str(i),'tupperware'+str(i),1)
    positive=e.sim.get_state();cases=[('one_vegetable_and_meat_each',None,True),('missing_meat','meat0',False),('missing_vegetable','vegetable1',False)]
   elif task=='CountertopCleanup':
    put('utensil',center(e.drawer));put('receptacle',center(e.cab))
    placed=False
    for points in e.fridge.get_int_sites(relative=False).values():
     p0,px,py,pz=points;put('food_bowl',p0+((px-p0)+(py-p0)+(pz-p0))/2)
     if e.fridge.check_rack_contact(e,'food_bowl',compartment='fridge'):placed=True;break
    if not placed:raise RuntimeError('No valid native fridge region for diagnostic bowl')
    contact('food1','food_bowl',-1);contact('food2','food_bowl',1)
    positive=e.sim.get_state();cases=[('all_sorted',None,True),('utensil_outside','utensil',False),('missing_food','food1',False),('bowl_outside_fridge','food_bowl',False)]
   elif task=='MicrowaveCorrectMeal':
    bowl='bowl'+str(e.target_bowl);put(bowl,center(e.microwave))
    contact('food0_'+bowl,bowl,-1);contact('food1_'+bowl,bowl,1)
    e.microwave.close_door(env=e);e.sim.forward();e.microwave._turned_on=True
    positive=e.sim.get_state();cases=[('correct_bowl_closed_on',None,True),('microwave_off','off',False),('door_open','door',False),('food_missing','food0_'+bowl,False),('target_bowl_outside',bowl,False)]
   else:
    e.dishwasher.close_door(env=e);e.sim.forward()
    joint=e.dishwasher._joint_names['rack'];bid=e.sim.model.jnt_bodyid[e.sim.model.joint_name2id(joint)]
    geoms=[i for i in range(e.sim.model.ngeom) if e.sim.model.geom_bodyid[i]==bid];origins=[e.sim.data.geom_xpos[i].copy() for i in geoms]
    for j,name in enumerate(('dish0','dish1')):
     placed=False
     for origin in origins:
      for dz in np.linspace(.25,-.10,176):
       put(name,origin+[(-1 if j==0 else 1)*.06,0,dz])
       if e.dishwasher.check_rack_contact(e,name):placed=True;break
      if placed:break
     if not placed:raise RuntimeError('No real rack collision found for '+name)
    positive=e.sim.get_state();cases=[('both_rack_closed',None,True),('one_dish_missing','dish0',False),('door_open','door',False)]
   writer=imageio.get_writer(str(a.output/(task+'-state-fixtures.mp4')),fps=20,codec='libx264',quality=8)
   for label,perturb,expected in cases:
    e.sim.set_state(positive);e.sim.forward()
    if task=='MicrowaveCorrectMeal':e.microwave._turned_on=perturb!='off'
    if perturb=='door':
     fixture=e.microwave if task=='MicrowaveCorrectMeal' else e.dishwasher;fixture.open_door(min=.7,max=.7,env=e);e.sim.forward()
    elif perturb and perturb!='off':put(perturb,pos(perturb)+[2,0,0])
    actual=bool(e._check_success());depths=[float(e.sim.data.contact[i].dist) for i in range(e.sim.data.ncon)]
    rows.append({'task':task,'fixture':label,'expected':expected,'actual':actual,'passed':actual==expected,'all_objects':{name:pos(name).tolist() for name in e.objects},'contact_count':e.sim.data.ncon,'minimum_contact_distance':min(depths,default=None),'dynamically_settled':False})
    for _ in range(15):writer.append_data(e.sim.render(camera_name=e.camera_names[0],width=640,height=480)[::-1])
  except Exception as exc:errors.append({'task':task,'error':type(exc).__name__+': '+str(exc)})
  finally:
   if writer:writer.close()
   if wrapper:wrapper.close()
 result={'seed':7,'split':'pretrain','kind':'manually set paused native contact/geometric states; not a rollout, stable placement proof or successful robot policy','model_calls':0,'tests':rows,'errors':errors,'passed':sum(r['passed'] for r in rows)}
 (a.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));raise SystemExit(0 if not errors and all(r['passed'] for r in rows) else 1)
if __name__=='__main__':main()
