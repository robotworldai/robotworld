"""Paused simulator-state fixtures, separate from robot rollouts and official scores."""
import argparse,json
from pathlib import Path
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import robocasa

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=[]
 for task in ('CloseDrawer','ResetCabinetDoors'):
  env=gym.make('robocasa/'+task,split='pretrain',seed=7)
  try:
   env.reset(seed=7);e=env.unwrapped.env
   fixtures=[e.drawer] if task=='CloseDrawer' else [e.cab,e.cab2,e.cab3]
   native_references=[f.name for f in fixtures]
   fixtures=list({f.name:f for f in fixtures}.values())
   camera=e.camera_names[0]
   writer=imageio.get_writer(str(a.output/(task+'-state-fixtures.mp4')),fps=20,codec='libx264',quality=8)
   base=e.sim.get_state()
   cases=[('all_closed',{},True)]
   for i in range(len(fixtures)):cases.append((f'fixture_{i}_open',{i:.5},False))
   try:
    for name,opened,expected in cases:
     e.sim.set_state(base)
     for i,fixture in enumerate(fixtures):
      value=opened.get(i,0.)
      if task=='ResetCabinetDoors':fixture.set_joint_state(min=value,max=value,env=e,joint_names=fixture.door_joint_names)
      else:fixture.set_door_state(min=value,max=value,env=e)
     e.sim.forward();actual=bool(e._check_success())
     row={'task':task,'fixture':name,'expected':expected,'actual':actual,'passed':actual==expected,'joint_qpos':e.sim.data.qpos.tolist(),'native_fixture_references':native_references,'fixture_types':[type(f).__name__ for f in fixtures],'door_states':[f.get_joint_state(e,f.door_joint_names) for f in fixtures] if task=='ResetCabinetDoors' else None};rows.append(row)
     for _ in range(30):writer.append_data(e.sim.render(camera_name=camera,width=640,height=480)[::-1])
   finally:writer.close();e.sim.set_state(base);e.sim.forward()
  finally:env.close()
 result={'seed':7,'split':'pretrain','kind':'manually placed simulator-state fixtures; 30 frozen render frames each; NOT a control rollout or task completion','model_calls':0,'tests':rows,'passed':sum(r['passed'] for r in rows)}
 (a.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));raise SystemExit(0 if all(x['passed'] for x in rows) else 1)
if __name__=='__main__':main()
