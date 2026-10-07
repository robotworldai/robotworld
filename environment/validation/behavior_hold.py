"""No-model native joint hold using the official evaluator stepping path."""
import json
import torch

def serial(value):
 if hasattr(value,'tolist'):return value.tolist()
 if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [serial(v) for v in value]
 if value is None or isinstance(value,(str,int,float,bool)):return value
 return str(value)

def run(evaluator,robot,output,steps):
 state=evaluator.instance_eval_states[0]
 action=torch.zeros((1,robot.action_dim),dtype=torch.float32)
 q=robot.get_joint_positions()
 for side in ('left','right'):
  action[0,robot.controller_action_idx['arm_'+side]]=q[robot.arm_control_idx[side]]
  action[0,robot.controller_action_idx['gripper_'+side]]=1.0
 action[0,robot.controller_action_idx['trunk']]=q[robot.trunk_control_idx]
 task=evaluator.env.task
 (output/'native-goals.json').write_text(json.dumps({'natural_language':serial(task.activity_natural_language_goal_conditions),'parsed':serial(task.compiled_task.conditions.parsed_goal_conditions),'initial_goal_status':serial(task._termination_conditions['predicate'].goal_status),'policy':'native absolute joint hold; not a solver; no model'},indent=2))
 # The evaluator's own writer preserves its sensor composition and native cadence.
 from omnigibson.eval.evaluator import create_video_writer
 video=output/'videos'/('hold-'+evaluator.cfg.task.name+'.mp4')
 evaluator._set_video_writer(state,create_video_writer(fpath=str(video),resolution=(448,672),rate=30))
 completed=0;terminated=truncated=False
 try:
  evaluator._write_video(state)
  with (output/'predicates.jsonl').open('w') as f:
   for _ in range(steps):
    term,trunc,info=evaluator._apply_actions(action,[0]);completed+=1
    evaluator._write_video(state)
    terminated=bool(term[0]);truncated=bool(trunc[0])
    f.write(json.dumps({'step':completed,'success':state.env_accessor.success,'terminated':terminated,'truncated':truncated,'info':serial(info),'goal_status':serial(task._termination_conditions['predicate'].goal_status)})+'\n');f.flush()
    if terminated or truncated:break
 finally:evaluator._set_video_writer(state,None)
 (output/'result.json').write_text(json.dumps({'task':evaluator.cfg.task.name,'control_steps':completed,'model_calls':0,'success':state.env_accessor.success,'terminated':terminated,'truncated':truncated,'diagnostic_budget_exhausted':not (terminated or truncated),'video_frames':completed+1,'positive_predicate_fixture_verified':False},indent=2))
