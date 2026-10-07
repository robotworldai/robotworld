"""Native MuJoCo predicates on explicit paused states; no model or policy score."""
import argparse,json,math
from pathlib import Path
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import robocasa

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=[]
 for task in ('NavigateKitchen','SortingCleanup'):
  wrapper=gym.make('robocasa/'+task,split='pretrain',seed=7)
  try:
   wrapper.reset(seed=7);e=wrapper.unwrapped.env;base=e.sim.get_state()
   writer=imageio.get_writer(str(a.output/(task+'-state-fixtures.mp4')),fps=20,codec='libx264',quality=8)
   if task=='NavigateKitchen':
    names=[n for n in e.sim.model.joint_names if 'joint_mobile_' in n];assert len(names)==3,names
    addresses=[e.sim.model.get_joint_qpos_addr(n) for n in names]
    bid=e.sim.model.body_name2id('mobilebase0_base')
    def pose():
     m=e.sim.data.body_xmat[bid].reshape(3,3)
     return np.r_[e.sim.data.body_xpos[bid][:2],math.atan2(m[1,0],m[0,0])]
    def wrap(v):
     v=v.copy();v[2]=(v[2]+np.pi)%(2*np.pi)-np.pi;return v
    def place(delta,yaw):
     desired=np.r_[e.target_pos[:2]+delta,e.target_ori[2]+yaw]
     for _ in range(8):
      e.sim.forward();old=pose();err=wrap(desired-old)
      if np.linalg.norm(err)<1e-8:break
      columns=[]
      for address in addresses:
       v=e.sim.data.qpos[address];e.sim.data.qpos[address]=v+1e-5;e.sim.forward();columns.append(wrap(pose()-old)/1e-5);e.sim.data.qpos[address]=v
      change=np.linalg.solve(np.array(columns).T,err)
      for address,value in zip(addresses,change):e.sim.data.qpos[address]+=value
     e.sim.forward();return {'base_pose':pose().tolist(),'target':desired.tolist(),'joint_names':names}
    cases=[('target',[0,0],0,True),('distance_199mm',[.199,0],0,True),('distance_201mm',[.201,0],0,False),('yaw_010rad',[0,0],.1,True),('yaw_021rad',[0,0],.21,False)]
   else:
    def center(f):
     p0,px,py,pz=next(iter(f.get_int_sites(relative=False).values()));return p0+((px-p0)+(py-p0)+(pz-p0))/2
    def place(label,opened):
     poses={}
     for name,f in [('mug',e.sink),('bowl',e.cab)]:
      pos=center(f)+(np.array([2,0,0]) if label==name+'_outside' else 0)
      e.sim.data.set_joint_qpos(e.objects[name].joints[0],np.r_[pos,[1,0,0,0]]);poses[name]=pos.tolist()
     e.cab.set_joint_state(min=opened,max=opened,env=e,joint_names=e.cab.door_joint_names);e.sim.forward();return poses
    cases=[('sorted_closed','sorted',0.,True),('sorted_open','sorted',.6,False),('mug_outside','mug_outside',0.,False),('bowl_outside','bowl_outside',0.,False)]
   try:
    for label,arg1,arg2,expected in cases:
     e.sim.set_state(base);e.sim.forward();state=place(arg1,arg2);actual=bool(e._check_success())
     rows.append({'task':task,'fixture':label,'expected':expected,'actual':actual,'passed':actual==expected,'state':state})
     for _ in range(15):writer.append_data(e.sim.render(camera_name=e.camera_names[0],width=640,height=480)[::-1])
   finally:writer.close();e.sim.set_state(base);e.sim.forward()
  finally:wrapper.close()
 result={'seed':7,'split':'pretrain','kind':'paused native simulator-state fixtures; not robot rollout or task completion','model_calls':0,'tests':rows,'passed':sum(r['passed'] for r in rows)}
 (a.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));raise SystemExit(0 if all(r['passed'] for r in rows) else 1)
if __name__=='__main__':main()
