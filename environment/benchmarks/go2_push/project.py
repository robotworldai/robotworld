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
        spec=importlib.util.spec_from_file_location('world_go2_push_isaac601',compat)
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
    (OUTPUT/'go2_push-compatibility.json').write_text(json.dumps({'runtime':runtime,'experimental':runtime=='6.0.1.0','upstream_files_modified':False,'play_config_used':False,'actor_group':'policy only; no critic'},indent=2))

def _namespace(name,path):
    if name not in sys.modules:
        module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module

def _packages():
    _namespace('isaaclab_go2_pushrecovery',SOURCE/'src/isaaclab_go2_pushrecovery')
    lab=SOURCE.parent/'isaaclab211'
    # Pin inherited task/robot config to upstream v2.1.1, not whichever version
    # happens to be in the runtime image. IsaacLab core remains experimental2.2.
    for base,relative,suffixes in [
        ('isaaclab_tasks','source/isaaclab_tasks/isaaclab_tasks',['','.manager_based','.manager_based.locomotion','.manager_based.locomotion.velocity','.manager_based.locomotion.velocity.config','.manager_based.locomotion.velocity.config.go2']),
        ('isaaclab_assets','source/isaaclab_assets/isaaclab_assets',['','.robots'])]:
        for suffix in suffixes:
            name=base+suffix;path=lab/relative
            if suffix:path=path.joinpath(*suffix.strip('.').split('.'))
            if name in sys.modules:sys.modules[name].__path__=[str(path)]
            else:_namespace(name,path)
    import subprocess
    if subprocess.check_output(['git','-c',f'safe.directory={lab}','-C',str(lab),'rev-parse','HEAD'],text=True).strip()!='90b79bb2d44feb8d833f260f2bf37da3487180ba':
        raise RuntimeError('Expected pinned original IsaacLab2.1.1 inherited task definitions')
    # The environment module also defines a PPO training config. Import its
    # original config classes without eagerly loading unused RSL VecEnv runners.
    rl_root=lab/'source/isaaclab_rl/isaaclab_rl'
    _namespace('isaaclab_rl',rl_root)
    _namespace('isaaclab_rl.rsl_rl',rl_root/'rsl_rl')
    rl_cfg=importlib.import_module('isaaclab_rl.rsl_rl.rl_cfg')
    for name in ['RslRlOnPolicyRunnerCfg','RslRlPpoActorCriticCfg','RslRlPpoAlgorithmCfg']:
        setattr(sys.modules['isaaclab_rl.rsl_rl'],name,getattr(rl_cfg,name))


def build(task_id,seed,output):
    if task_id not in ['T10','Isaac-Velocity-Flat-Unitree-Go2-PushRecovery-v0']:raise ValueError(task_id)
    module=importlib.import_module('isaaclab_go2_pushrecovery.env_cfg')
    cfg=module.UnitreeGo2PushRecoveryEnvCfg()
    cfg.scene.num_envs=1;cfg.seed=seed;cfg.wait_for_textures=False
    asset=SOURCE.parent/'assets/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd'
    manifest=SOURCE.parent/'asset-manifest.json'
    if not manifest.exists() or not json.loads(manifest.read_text()).get('complete'):
        raise RuntimeError('Prepare complete official Go2 USD dependency closure before running')
    cfg.scene.robot.spawn.usd_path=str(asset)
    _localize_debug_markers(cfg)
    return cfg


def metrics(env):
    if getattr(env.cfg,'world_scoring_profile',None)=='world-state-v1':
        return {'success':None,'original_push_schedule_replaced':True,
                'push_protocol':'See world-scene.json and world_scene_events; World 30 N scheduled pulses'}
    push=importlib.import_module('isaaclab_go2_pushrecovery.push_events')
    count=lambda state: int(state.trigger_count.sum().item()) if state.initialized else 0
    return {'success':None,'success_definition':'Native velocity tracking and fall termination; no published random-push binary SR',
        'common_step_counter':int(env.common_step_counter),
        'impulse_trigger_count':count(push._impulse_state),
        'sustained_trigger_count':count(push._sustained_state),
        'impulse_max_force_N':float(env.event_manager.get_term_cfg('push_robot_impulse').params['max_force']),
        'sustained_max_force_N':float(env.event_manager.get_term_cfg('push_robot_sustained').params['max_force'])}


def instruction(task_id,env=None):
    return '''Control Unitree Go2 full-body joint targets to follow the original sampled body velocity/yaw commands and reject native random force pulses. No pretrained gait or whole-body stabilizer is provided: only native DC motor PD tracking.
Native action is12 joint-position residuals: target=default_joint_position+0.25*u radians. The actual articulation joint order is exported in action metadata; use that exact order rather than assuming per-leg order. Default hips: left+0.1/right-0.1rad, front thighs0.8/rear thighs1.0, calves-1.5. Motor stiffness25, damping0.5, saturation23.5Nm. Native term has no configured action clipping; physical limits still apply. Do not confuse a nominal normalized value with a guaranteed safe joint range.
Native policy terms are body linear velocity(3), angular velocity(3), projected gravity(3), velocity_commands(3), joint position relative to default(12), joint velocity(12), last action(12):48values with original observation noise/scaling. No critic data or terrain height scan is added. API delivers named native_policy_terms; inspect metadata and names. Native state observation, not claimed pure visual.
One control tick=.02s, four physics substeps=.005s; horizon20s/1000ticks. Preserve the sampled commands, reset randomization and original noisy actor input. A code-control function can provide per-tick feedback from the delivered actor terms; it cannot read the environment or hidden force state directly.
Random impulse triggers every6-10s with requested duration0.15-0.25s; original event checks every0.18-0.22s, so actual force expiry is quantized to checks. Impulse max_force starts30N, not120N. Curriculum reaches120N after19200common steps; do not pretend a fresh single episode is the trained end curriculum. Sustained loads are scheduled25-40s apart, requested8-12s duration, start10N and rise later; these may not occur in a20s episode. Do not claim sustained-load recovery if none triggered. This is not the author's scheduled120N demo.
Report native reward/velocity tracking, base-contact termination, sampled-force curriculum state and actual trigger counts. Surviving the horizon alone is not a task success rate. Continue attempting control until a native termination or budget limit. Third-person review video does not grant privileged policy observations.'''


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
        if str(getattr(value, 'texture_file', '')).endswith('/Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr'):
            texture = Path(__file__).resolve().parents[3] / 'third_party/benchmarks/go2_push/assets/Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr'
            if not texture.is_file():raise FileNotFoundError(texture)
            value.texture_file = str(texture)
        children = value.values() if isinstance(value, dict) else value if isinstance(value, (list, tuple)) else vars(value).values() if hasattr(value, '__dict__') else ()
        for child in children:visit(child)
    visit(cfg)
