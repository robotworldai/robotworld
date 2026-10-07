"""Actual MuJoCo geometry predicates at manually placed states, never policy scores."""
import argparse,json,math
from pathlib import Path
import gymnasium as gym
import robocasa
import imageio.v2 as imageio
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=[]
 for task in ('CoffeeSetupMug','OrganizeMugsByHandle'):
  wrapper=gym.make('robocasa/'+task,split='pretrain',seed=7)
  try:
   wrapper.reset(seed=7);e=wrapper.unwrapped.env;base=e.sim.get_state()
   name='obj' if task=='CoffeeSetupMug' else 'mug_counter1';joint=e.objects[name].joints[0]
   if task=='CoffeeSetupMug':
    site=e.coffee_machine.naming_prefix+'receptacle_place_site';center=e.sim.data.site_xpos[e.sim.model.site_name2id(site)].copy();yaw=0
    cases=[('under_spout',[0,0,0],0,True),('xy_39mm',[.039,0,0],0,True),('xy_41mm',[.041,0,0],0,False),('height_99mm',[0,0,.099],0,True),('height_101mm',[0,0,.101],0,False)]
   else:
    p0,px,py,pz=next(iter(e.cabinet.get_int_sites(relative=False).values()));center=p0+((px-p0)+(py-p0)+(pz-p0))/2;yaw=e.cabinet.rot
    cases=[('inside_handle_right',[0,0,0],90,True),('inside_handle_wrong',[0,0,0],270,False),('outside_cabinet',[2,0,0],90,False)]
   writer=imageio.get_writer(str(a.output/(task+'-state-fixtures.mp4')),fps=20,codec='libx264',quality=8)
   try:
    for label,delta,degrees,expected in cases:
     e.sim.set_state(base);angle=yaw+math.radians(degrees);quat=[math.cos(angle/2),0,0,math.sin(angle/2)]
     pos=center+np.array(delta);e.sim.data.set_joint_qpos(joint,np.r_[pos,quat]);e.sim.forward()
     actual=bool(e._check_success());rows.append({'task':task,'fixture':label,'expected':expected,'actual':actual,'passed':actual==expected,'position':pos.tolist(),'quaternion_wxyz':quat})
     for _ in range(15):writer.append_data(e.sim.render(camera_name=e.camera_names[0],width=640,height=480)[::-1])
   finally:writer.close();e.sim.set_state(base);e.sim.forward()
  finally:wrapper.close()
 result={'seed':7,'split':'pretrain','kind':'native simulator-state geometry fixtures, frozen render frames; NOT a control rollout or successful policy','model_calls':0,'tests':rows,'passed':sum(r['passed'] for r in rows)}
 (a.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));raise SystemExit(0 if all(r['passed'] for r in rows) else 1)
if __name__=='__main__':main()
