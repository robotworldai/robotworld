"""Unmodified native sequence / geometry predicates on explicit synthetic states."""
import argparse,copy,importlib,itertools,json,math
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
 piano=importlib.import_module('success.custom.task_79_bimanual_piano_melody')
 glass=importlib.import_module('success.custom.task_03_wine_glass_plate_balance')
 def check(task,label,actual,expected):rows.append({'task':task,'fixture':label,'actual':bool(actual),'expected':expected,'passed':bool(actual)==expected})
 def play(extra=False,left_only=False,release=True):
  state={'obj_325_piano_1':{'pose_world':[0,0,0,0,0,0,1],'qpos':{}}};ctx={};q=state['obj_325_piano_1']['qpos'];actual=False
  for left,right in zip(piano._LEFT_MELODY,piano._RIGHT_MELODY):
   if extra:q['joint_19_to_1']=.005;piano.check_success(state,ctx,{})
   if release:q.clear();piano.check_success(state,ctx,{})
   q[f'joint_{left}_to_1']=.005
   if not left_only:q[f'joint_{right}_to_1']=.005
   actual=piano.check_success(state,ctx,{})
  return actual,ctx
 check('41','both_complete',play()[0],True)
 check('41','left_hand_only',play(left_only=True)[0],False)
 check('41','held_notes_do_not_count_repetition',play(release=False)[0],False)
 check('41','extra_wrong_notes_are_ignored_by_native',play(extra=True)[0],True)
 check('41','missing_piano',piano.check_success({}, {}, {}),False)
 q={'joint_18_to_1':.004};state={'obj_325_piano_1':{'pose_world':[0,0,0,0,0,0,1],'qpos':q}};ctx={};piano.check_success(state,ctx,{})
 check('41','press_threshold_strict',ctx['left_index']==0,True)
 def states():return {name:{'pose_world':[x,y,.9,math.sqrt(.5),0,0,math.sqrt(.5)],'lin_vel_world':[0,0,0],'ang_vel_world':[0,0,0]} for name,(x,y) in zip(glass.GLASS_IDS,glass.TARGETS)}
 for perm in itertools.permutations(glass.TARGETS):
  s=states()
  for name,(x,y) in zip(glass.GLASS_IDS,perm):s[name]['pose_world'][:2]=[x,y]
  check('49','any_unique_target_assignment:'+str(perm),glass.check_success(s,{},{}),True)
 for label,mutate,expected in [
  ('duplicate_target',lambda s:s[glass.GLASS_IDS[1]]['pose_world'].__setitem__(slice(0,2),list(glass.TARGETS[0])),False),
  ('outside_101mm',lambda s:s[glass.GLASS_IDS[0]]['pose_world'].__setitem__(0,glass.TARGETS[0][0]+.101),False),
  ('inside_99mm',lambda s:s[glass.GLASS_IDS[0]]['pose_world'].__setitem__(0,glass.TARGETS[0][0]+.099),True),
  ('moving_fast',lambda s:s[glass.GLASS_IDS[0]].__setitem__('lin_vel_world',[.051,0,0]),False),
  ('wrong_upright_axis',lambda s:s[glass.GLASS_IDS[0]]['pose_world'].__setitem__(slice(3,7),[0,0,0,1]),False),
  ('height_not_in_native_terminal',lambda s:s[glass.GLASS_IDS[0]]['pose_world'].__setitem__(2,2.0),True)]:
  s=states();mutate(s);check('49',label,glass.check_success(s,{},{}),expected)
 result={'kind':'actual upstream predicate functions; synthetic input states, not simulator physical fixtures','tests':rows,'total':len(rows),'passed':sum(r['passed'] for r in rows)}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps({'total':len(rows),'passed':result['passed']}));raise SystemExit(0 if all(r['passed'] for r in rows) else 1)
if __name__=='__main__':main()
