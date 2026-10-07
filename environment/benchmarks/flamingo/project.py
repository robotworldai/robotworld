"""Native task adapter; all compatibility lives outside unchanged checkout."""
from pathlib import Path
import importlib
import importlib.util
import importlib.metadata
import json
import sys
import types

SOURCE = None
OUTPUT = None

def setup(source: Path, output: Path):
    global SOURCE, OUTPUT
    source=Path(source); SOURCE=source/'checkout' if (source/'checkout').exists() else source; OUTPUT=Path(output)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    runtime=importlib.metadata.version('isaacsim')
    if runtime=='6.0.1.0':
        compat=SOURCE.parent.parent/'robolab/compat/isaac601.py'
        spec=importlib.util.spec_from_file_location('world_flamingo_isaac601',compat)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.install()
        root=SOURCE.parent.parent/'wheeledlab/assets'
        ground=root/'Isaac/Environments/Grid/default_environment.usd'
        if not ground.is_file():raise FileNotFoundError(ground)
        import carb
        carb.settings.get_settings().set('/persistent/isaac/asset_root/cloud',str(root))
        from isaaclab.sim import GroundPlaneCfg
        # configclass captures its default in generated __init__; assigning
        # GroundPlaneCfg.usd_path alone leaves instances at None/Isaac/....
        original_ground_init = GroundPlaneCfg.__init__
        def local_ground_init(self, *args, **kwargs):
            original_ground_init(self, *args, **kwargs)
            if str(self.usd_path).endswith('/Isaac/Environments/Grid/default_environment.usd'):
                self.usd_path = str(ground)
        GroundPlaneCfg.__init__ = local_ground_init
        # Native command debug markers may have been instantiated before Kit's
        # asset root existed. Preserve the exact official marker USD locally.
        from isaaclab.markers import config as marker_configs
        arrow = ground.parents[2] / 'Props/UIElements/arrow_x.usd'
        for marker_cfg in vars(marker_configs).values():
            markers = getattr(marker_cfg, 'markers', {})
            if not isinstance(markers, dict):continue
            for marker in markers.values():
                if str(getattr(marker, 'usd_path', '')).endswith('/Isaac/Props/UIElements/arrow_x.usd'):
                    if not arrow.is_file():raise FileNotFoundError(arrow)
                    marker.usd_path = str(arrow)
    _packages()
    (OUTPUT/'flamingo-compatibility.json').write_text(json.dumps({'runtime':runtime,'experimental':runtime=='6.0.1.0','upstream_files_modified':False,'play_config_used':False,'actor_group':'policy only; no critic'},indent=2))

def _namespace(name,path):
    if name not in sys.modules:
        module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module

policy_groups=['stack_policy','none_stack_policy']
NATIVE_POLICY_STACKS=2  # upstream train.py/play.py CLI default;3frames including current


def _packages():
    for name in ['lab','lab.flamingo','lab.flamingo.assets','lab.flamingo.assets.flamingo',
                 'lab.flamingo.tasks','lab.flamingo.tasks.manager_based','lab.flamingo.tasks.manager_based.locomotion',
                 'lab.flamingo.tasks.manager_based.locomotion.velocity',
                 'lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_env',
                 'lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_env.flat_env',
                 'lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_env.flat_env.track_jump']:
        _namespace(name,SOURCE.joinpath(*name.split('.')))
    # Same original computed constant; bypass unrelated robot re-export imports.
    sys.modules['lab.flamingo.assets.flamingo'].FLAMINGO_ASSETS_DATA_DIR=str(SOURCE/'lab/flamingo/assets/data')


def build(task_id,seed,output):
    if task_id not in ['T15','Isaac-TrackJUMP-Flat-Flamingo-v1-ppo']:raise ValueError(task_id)
    module=importlib.import_module('lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_env.flat_env.track_jump.flat_env_track_jump_cfg')
    cfg=module.FlamingoFlatEnvCfg()
    cfg.scene.num_envs=1;cfg.seed=seed;cfg.wait_for_textures=False
    asset=SOURCE.parent/'assets/Robots/Flamingo/flamingo_rev01_5_2/flamingo_rev01_5_2_merge_joints.usd'
    if not asset.is_file():raise FileNotFoundError('Run flamingo/prepare_assets.py: '+str(asset))
    cfg.scene.robot.spawn.usd_path=str(asset)
    _localize_debug_markers(cfg)
    # Upstream TrackJump removes the height sensor but leaves its critic term.
    # The actor never reads critic; omit the unused critic groups so Manager
    # construction does not require the removed height/lift sensors.
    if getattr(cfg.scene,'height_scanner',None) is None:
        for group_name in ('none_stack_critic','stack_critic'):
            setattr(cfg.observations,group_name,None)
    return cfg


def policy_observation(env,obs):
    """Use upstream StateHandler's current+2history convention, never critic."""
    import math
    stack,nonstack=obs['stack_policy'],obs['none_stack_policy']
    if not hasattr(env,'_world_actor_handler'):
        path=SOURCE/'scripts/co_rl/core/utils/state_handler.py'
        spec=importlib.util.spec_from_file_location('world_flamingo_original_state_handler',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        env._world_actor_handler=module.StateHandler(NATIVE_POLICY_STACKS+1,stack.shape[-1],nonstack.shape[-1])
    handler=env._world_actor_handler
    first=int(env.episode_length_buf[0].item())==0
    vector=handler.reset(stack,nonstack) if first else handler.update(stack,nonstack)
    values=vector[0].detach().cpu().tolist()
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite native Flamingo actor input')
    return {'actor_vector':values,'stack_policy_latest':stack[0].detach().cpu().tolist(),
            'none_stack_policy_current':nonstack[0].detach().cpu().tolist()}


def metrics(env):
    return {'success':None,'success_definition':'Native jump-event upward-velocity/push-ground/tracking rewards and illegal contact; no binary jump+stable-landing SR',
        'native_event_command':env.command_manager.get_command('event')[0].detach().cpu().tolist(),
        'native_horizon_seconds':20.,'policy_history_frames':NATIVE_POLICY_STACKS+1,
        'policy_history_source':'Original co_rl StateHandler, train/play CLI default num_policy_stacks=2, current first'}


def instruction(task_id,env=None):
    return '''Control the original Flamingo rev01_5_2 wheel-legged robot to follow native velocity commands, jump when its native event command is active, land and continue balancing. No pretrained jump/gait policy is supplied. Native low-level delayed PD/gear actuator remains; it does not choose the whole-body jump or balance sequence.
There are exactly8native actions in this order: [left_hip,right_hip,left_shoulder,right_shoulder,left_leg,right_leg,left_wheel,right_wheel]. The first6are absolute actuator position targets with scale1rad and no default offset. The final2are wheel velocity commands with scale40rad/s and no offset. Important: leg joints use the ORIGINAL GearDelayedPDActuator, gear_ratio=-1.5: measured physical leg position/velocity is multiplied by-1.5 before motor-space PD, and torque is mapped back by gear ratio. Thus leg target is motor-space; do not treat it as direct physical leg-joint angle. Action term does not declare an explicit clip; motor/joint limits remain. Delays randomly span0-4physical ticks (0-20ms); keep feedback stable.
The allowed native actor groups are stack_policy and none_stack_policy, never their critic counterparts. stack_policy is28values: hip+shoulder positions4, leg motor-space positions2, hip+shoulder velocities4(scale.15), leg motor-space velocities2(scale.15), wheel velocities2(scale.15), base angular velocity3(scale.25), projected gravity3, previous action8. Within each regex-resolved term use runtime metadata for joint order. Original noise remains active. none_stack_policy contains velocity_commands4 (vx,vy,yaw,height; first3 scaled[1,0,.25],height unscaled) and event_commands2(active flag,elapsed time). For TrackJump base linear velocity,base height,height scan,current reward,contact flag,lift mask,roll/pitch commands are removed from actor.
The API provides actor_vector using the ORIGINAL co_rl StateHandler with current+2previous stack_policy frames (newest first), followed by current none_stack_policy: typically90values. Initial history repeats the first frame, exactly the native handler. It additionally labels stack_policy_latest and none_stack_policy_current; they contain no new privileged data. Native config class default stacks0 differs from author train/play CLI default2; this adapter explicitly uses CLI default2 and records it.
Native physics200Hz, control50Hz (.02s/action), horizon20s=1000steps. Jump event resamples3-5s, lasts1.2s, first activates only after native2s warmup;10% standing events suppress jumps. Native reward emphasizes upward velocity/ground push during event elapsed0.3-0.8s, wheel quietness in event, orientation, smoothness and command tracking. Velocity command resamples9-13s; vx[-1.5,1.5]m/s,yaw[-2.5,2.5]rad/s. Preserve these commands rather than overwriting them.
A native velocity push arrives13-15s: x[-1.5,1.5],y[-1,1],z[-1,.5]m/s. Mass/friction/reset perturbations stay enabled. Base,hip,shoulder,leg unwanted contacts can terminate; this is NOT Play, whose push magnitude is zero. There is no upstream full jump+stable-landing binary success; evaluator reports original reward terms/termination and event command traces, not survival-as-success.
For quick takeoff/landing/balance feedback consider coding_control if enabled: read only provided actor data each control tick and return the same8Daction. Code cannot query simulator truth or modify scene, command,physics,reset or scoring. Continue until native termination or budget. Third-person review rendering is for the human only.'''


def _localize_debug_markers(cfg):
    """Resolve copied native marker configs; change only official asset location."""
    arrow = Path(__file__).resolve().parents[3] / 'third_party/benchmarks/wheeledlab/assets/Isaac/Props/UIElements/arrow_x.usd'
    seen = set()
    def visit(value):
        if id(value) in seen or isinstance(value, (str, int, float, bool, bytes, type, type(None))) or callable(value):
            return
        seen.add(id(value))
        if str(getattr(value, 'usd_path', '')).endswith('/Isaac/Props/UIElements/arrow_x.usd'):
            if not arrow.is_file():raise FileNotFoundError(arrow)
            value.usd_path = str(arrow)
        children = value.values() if isinstance(value, dict) else value if isinstance(value, (list, tuple)) else vars(value).values() if hasattr(value, '__dict__') else ()
        for child in children:visit(child)
    visit(cfg)
