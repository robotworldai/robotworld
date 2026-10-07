"""External registration/client entry point; delegates scoring and stepping to RoboLab."""
import argparse,json
from pathlib import Path
import cv2
from isaaclab.app import AppLauncher
from robolab.eval.runner import add_common_eval_args,run_evaluation
p=argparse.ArgumentParser();add_common_eval_args(p);AppLauncher.add_app_launcher_args(p)
p.add_argument('--isaac601-compat',action='store_true');p.add_argument('--manifest',required=True);p.add_argument('--control-mode',choices=['joint_position','absolute_ik','relative_ik'],default='joint_position')
p.add_argument('--world-timeout',type=float,default=7200);p.add_argument('--max-tool-calls',type=int,default=5000)
p.add_argument('--world-step-cap',type=int,help='Optional shorter rollout horizon; native success/failure predicates retained')
p.add_argument('--world-seed',type=int)
a=p.parse_args()
if a.num_envs!=1 or a.enable_gt_state:p.error('World supports one env and no privileged GT channel')
if a.num_runs!=1 or a.num_episodes_adaptive is not None:p.error('Run isolated single episodes per invocation; aggregate official results afterwards')
if not a.task or len(a.task)!=1:p.error('Select one explicit task per isolated invocation')
a.enable_cameras=True
if a.isaac601_compat:
    import isaacsim
    a.experience=str(Path(isaacsim.__file__).resolve().parent/'apps/isaacsim.exp.full.kit')
launcher=AppLauncher(a);app=launcher.app
try:
    if a.isaac601_compat:
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'third_party/benchmarks/robolab/compat'))
        from isaac601 import install
        install(exact_contact_paths=True)
    import robolab.constants as constants
    constants.ENABLE_SUBTASK_PROGRESS_CHECKING=a.enable_subtask
    if a.control_mode=='joint_position':
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs as register
    elif a.control_mode=='absolute_ik':
        from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs as register
    else:
        from robolab.registrations.droid.auto_env_registrations_rel_ik import auto_register_droid_rel_ik_envs as register
    register(task_dirs=a.task_dirs,task=a.task)
    from environment.benchmarks.robolab.policy import CodexClient
    Path('/runs/evaluation-config.json').write_text(json.dumps(vars(a),indent=2))
    def factory(args):return CodexClient(args.manifest,Path('/runs/agent')/args.task[0],args.control_mode,args.world_timeout,args.max_tool_calls)
    from environment.benchmarks.robolab.lifecycle import install_abort_cleanup
    install_abort_cleanup('/runs')
    from environment.benchmarks.robolab.rollout_config import install as install_rollout_config
    install_rollout_config('/runs',a.world_step_cap,a.world_seed)
    run_evaluation(a,policy='world_source_codex',client_factory=factory)
    Path('/runs/evaluation-finished.json').write_text(json.dumps({'official_runner_completed':True}))
except BaseException:
 import traceback
 error=traceback.format_exc();Path('/runs/error.txt').write_text(error);print(error,flush=True)
 raise
finally:app.close()
