"""External BEHAVIOR v3.9.3 official-evaluator entry point; upstream files are unchanged."""
import argparse
import json
from pathlib import Path


def robot_profile(robot, config, observation_only=False):
    fields = {}
    for name in config['proprio_obs']:
        if name == 'base_qvel': size = len(robot.base_control_idx)
        elif name.startswith('eef_'): size = 4 if name.endswith('_quat') else 3
        elif name.startswith('arm_'): size = len(robot.arm_control_idx[name.split('_')[1]])
        elif name.startswith('gripper_'): size = len(robot.gripper_control_idx[name.split('_')[1]])
        elif name.startswith('trunk_'): size = len(robot.trunk_control_idx)
        else: raise ValueError(f'Proprio field not approved: {name}')
        fields[name] = size
    idx = {name: value.tolist() for name,value in robot.controller_action_idx.items()}
    allowed = {'base','arm_left','arm_right','gripper_left','gripper_right','trunk','camera'}
    if set(idx)-allowed: raise ValueError('Unsupported controller layout')
    arm_size = 7 if observation_only and config['controller_config']['arm_left']['name'] == 'JointController' else 6
    for name,size in [('base',3),('arm_left',arm_size),('arm_right',arm_size),('gripper_left',1),('gripper_right',1)]:
        if len(idx.get(name,[]))!=size: raise ValueError(f'Unexpected {name} action dimension')
    if idx.get('camera'): raise ValueError('Expected null camera controller')
    if sum(fields.values())!=robot.proprioception_dim: raise ValueError('Proprio schema mismatch')
    trunk_limits = robot.control_limits['position']
    trunk_idx = robot.trunk_control_idx
    return {'trunk_joint_names':list(robot.trunk_joint_names), 'trunk_limits':[trunk_limits[0][trunk_idx].tolist(),trunk_limits[1][trunk_idx].tolist()],
            'action_dim':robot.action_dim,'controller_indices':idx,'proprio_fields':fields,
            'proprio_key':robot.name+'::proprio',
            'camera_keys':{k:robot.name+'::'+v for k,v in config['eval']['camera_sensor_names'].items()}}


def main():
    import faulthandler, signal
    faulthandler.enable()
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=False)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--isaac601-compat',action='store_true')
    parser.add_argument('--probe-only',action='store_true',help='Load/reset official scene and capture onboard observations; no model or action')
    parser.add_argument('--probe-steps',type=int,default=0)
    parser.add_argument('--task-name',default='carrying_in_groceries')
    parser.add_argument('--instance-index',type=int,default=10,help='Public split index; 10-19 for development')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--model-catalog',type=Path)
    parser.add_argument('--model')
    parser.add_argument('--robot-config',type=Path,help='Probe defaults to the upstream robot; model runs use the World IK tool profile')
    parser.add_argument('--disable-coding-control',action='store_true')
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--max-actions',type=int,default=100)
    parser.add_argument('--timeout',type=float,default=1800)
    args=parser.parse_args()
    if not 0<=args.instance_index<20: parser.error('Public instance index must be 0..19')
    if args.max_actions<=0 or args.timeout<=0 or (args.max_steps is not None and args.max_steps<=0): parser.error('Budgets must be positive')
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    if args.isaac601_compat:
        from environment.benchmarks.behavior_1k.compat import activate
        activate(out)
    from omegaconf import OmegaConf
    from omnigibson.eval.evaluator import BatchedEvaluator, resolve_instance_ids
    from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything
    from omnigibson.macros import gm
    gm.HEADLESS=True; gm.RENDER_VIEWER_CAMERA=False
    seed=seed_everything(DEFAULT_EVAL_SEED)
    if args.robot_config is None:
        args.robot_config = (Path('/behavior-src/OmniGibson/omnigibson/eval/r1pro.yaml') if args.probe_only
                             else Path(__file__).resolve().parents[1]/'robots/r1pro/behavior_ik.yaml')
    config=OmegaConf.to_container(OmegaConf.load(args.robot_config),resolve=True)
    # Keep the first integration tied to the reviewed IK contract.
    for side in ('left','right'):
        arm=config['controller_config'][f'arm_{side}']
        if not args.probe_only and (arm['name']!='InverseKinematicsController' or arm.get('mode')!='absolute_pose'):
            raise ValueError('This adapter requires absolute-pose IK controllers')
    instruction_file=Path('/behavior-src/docs/challenge/task_data.json')
    if not instruction_file.exists(): instruction_file=Path(__file__).resolve().parents[2]/'third_party/benchmarks/behavior_1k/checkout/docs/challenge/task_data.json'
    task_data=json.loads(instruction_file.read_text())
    items=task_data if isinstance(task_data,list) else task_data['tasks']
    task_entry=next(t for t in items if t['id']==args.task_name)
    task_instruction=task_entry.get('instruction') or task_entry.get('name') or args.task_name.replace('_',' ')
    instruction_missing=not bool(task_entry.get('instruction'))
    ids=resolve_instance_ids(args.task_name,[args.instance_index],mode='public_test')
    cfg=OmegaConf.create({'env_wrapper':{'_target_':'omnigibson.eval.wrappers.RGBDFullResWrapper'},
        'policy_name':'WorldCodex','model':{'_target_':'environment.benchmarks.behavior_1k.policy.WorldPolicy',
        'output_dir':str(out),'manifest':str(args.manifest.resolve()),'task_name':args.task_name,'task_instruction':task_instruction,
        'max_actions':args.max_actions,'timeout':args.timeout,'model':args.model,
        'step_budget':args.max_steps, 'coding_control_enabled':not args.disable_coding_control and not args.probe_only,
        'model_catalog':str(args.model_catalog.resolve()) if args.model_catalog else None},
        'headless':True,'partial_scene_load':True,'max_steps':args.max_steps,'write_video':True,
        'mode':'public_test','seed':seed,'num_envs':1,'task':{'name':args.task_name},'robot':config})
    (out/'evaluation-config.json').write_text(json.dumps({'task':args.task_name,'instance_index':args.instance_index,
        'instance_ids':ids,'max_steps_override':args.max_steps,'standard_step_budget':args.max_steps is None,
        'num_rollouts':1,'track':'onboard_rgb_depth_proprio','source_version':'v3.9.3','robot_config':config,
        'probe_only':args.probe_only,'coding_control_enabled':not args.disable_coding_control and not args.probe_only,'robot_config_path':str(args.robot_config),
        'isaac601_external_compat':args.isaac601_compat,'official_runtime':not args.isaac601_compat,
        'entrypoint':('official BatchedEvaluator._apply_actions via no-model diagnostic' if args.probe_only
                      else 'official BatchedEvaluator.run via external local policy')},indent=2))
    for name in ('json','videos'): (out/name).mkdir(exist_ok=True)
    (out/'progress.json').write_text(json.dumps({'phase':'initializing_official_evaluator'}))
    with BatchedEvaluator(cfg) as evaluator:
        policy=evaluator.policy
        if policy.step_budget is None:
            from omnigibson.eval.utils.eval_utils import EVAL_TIMEOUT_MULTIPLIER
            policy.step_budget=int(evaluator.human_stats['length']*EVAL_TIMEOUT_MULTIPLIER)
        if instruction_missing:
            # Some official website metadata (e.g. setting_the_table) omits
            # instruction. Use the actual task's public goal text, never
            # invent a replacement task or alter the upstream metadata.
            goals=list(evaluator.env.task.activity_natural_language_goal_conditions)
            if not goals:raise ValueError('Native task has no instruction or goal description')
            policy.task_instruction=task_instruction+'\nNative task goal conditions:\n'+'\n'.join(str(x) for x in goals)
            (out/'instruction-source.json').write_text(json.dumps({'source':'official task BDDL natural language goals','website_metadata_missing_instruction':True,'instruction':policy.task_instruction},indent=2))
        robot=evaluator.instance_eval_states[0].env_accessor.robot
        policy.bind(robot_profile(robot,config,observation_only=args.probe_only))
        (out/'progress.json').write_text(json.dumps({'phase':'resetting_scene'}))
        if args.probe_only:
            evaluator.load_batch({0:int(ids[0])},write_video=False)
            if args.isaac601_compat:
                from environment.benchmarks.behavior_1k.compat.render_diagnostics import snapshot, capture_sequence
                snapshot(out)
                capture_sequence(robot, config, out)
            if args.probe_steps:
                from environment.validation.behavior_hold import run
                run(evaluator,robot,out,args.probe_steps)
            policy._ingest(evaluator._batch_obs())
            policy._content()
            policy._save()
            (out/'progress.json').write_text(json.dumps({'phase':'probe_complete','actions_executed':args.probe_steps}))
            return
        (out/'progress.json').write_text(json.dumps({'phase':'running_source_codex'}))
        results=None
        from environment.runtime.nonaction_budget import NonActionBudgetExceeded
        # The upstream run() does not pass max_steps to its batch loop. Apply the
        # existing configured cap after the last real action, before another model
        # request, while retaining its native record_fn and success evaluation.
        from environment.runtime.control_step_cap import bounded_step
        evaluator._step_fn = bounded_step(evaluator._step_fn, policy.step_budget)
        try:
            results=evaluator.run([int(i) for i in ids],write_video=True,video_path=str(out/'videos'),
                                  metrics_dir=str(out/'json'),rollout_id=0,video_fps=30)
        except NonActionBudgetExceeded as stop:
            # Snapshot the same official metrics without taking another physics step.
            state=evaluator.instance_eval_states[0]
            result={'task':args.task_name,'instance_id':int(ids[0]),'rollout_id':0,
                    'steps':policy.step,'success':state.env_accessor.success,
                    'stop_reason':stop.snapshot['stop_reason'],'interaction_budget':stop.snapshot}
            for metric in state.metrics:result.update(metric.aggregate())
            results={int(ids[0]):result}
            policy.stopped=stop.snapshot['stop_reason']
            evaluator._set_video_writer(state,None)
        finally:
            # Do not expose evaluator metrics or simulator-only states as policy input.
            if hasattr(policy,'result'):
                final_obs=evaluator._batch_obs() if results is not None else None
                policy.finish(final_obs,results[int(ids[0])] if results is not None else None)
        (out/'results.json').write_text(json.dumps(results,indent=2,default=float))
        (out/'progress.json').write_text(json.dumps({'phase':'episode_saved'}))


if __name__=='__main__': main()
