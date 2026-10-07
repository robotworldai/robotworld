"""Native task adapter; all compatibility lives outside unchanged checkout."""
from pathlib import Path
import importlib
import importlib.util
import importlib.metadata
import json
import os
import sys
import types

SOURCE = None
OUTPUT = None
RUNTIME = None

def setup(source: Path, output: Path):
    global SOURCE, OUTPUT, RUNTIME
    source=Path(source); SOURCE=source/'checkout' if (source/'checkout').exists() else source; OUTPUT=Path(output)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    runtime=importlib.metadata.version('isaacsim')
    RUNTIME=runtime
    if runtime=='6.0.1.0':
        compat=SOURCE.parent.parent/'robolab/compat/isaac601.py'
        spec=importlib.util.spec_from_file_location('world_robot_lab_isaac601',compat)
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
    (OUTPUT/'robot_lab-compatibility.json').write_text(json.dumps({'runtime':runtime,'experimental':runtime=='6.0.1.0','upstream_files_modified':False,'play_config_used':False,'actor_group':'policy only; no critic'},indent=2))

def _namespace(name,path):
    if name not in sys.modules:
        module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module

def _packages():
    root=SOURCE/'source/robot_lab/robot_lab'
    for suffix in ['', '.tasks', '.tasks.locomotion', '.tasks.locomotion.velocity', '.tasks.locomotion.velocity.config', '.tasks.locomotion.velocity.config.quadruped', '.tasks.locomotion.velocity.config.quadruped.unitree_a1_handstand']:
        _namespace('robot_lab'+suffix,root.joinpath(*suffix.strip('.').split('.')) if suffix else root)


def build(task_id,seed,output):
    if task_id not in ['T11','RobotLab-Isaac-Velocity-Flat-HandStand-Unitree-A1-v0']:raise ValueError(task_id)
    module=importlib.import_module('robot_lab.tasks.locomotion.velocity.config.quadruped.unitree_a1_handstand.flat_env_cfg')
    cfg=module.UnitreeA1HandStandFlatEnvCfg()
    cfg.scene.num_envs=1;cfg.seed=seed;cfg.wait_for_textures=False
    # Fixed source sets10s, despite the materials table saying20s. Preserve it.
    if cfg.episode_length_s != 10.0:raise RuntimeError('Unexpected pinned handstand horizon')
    _localize_debug_markers(cfg)
    if os.environ.get('WORLD_A1_REIMPORT') == '1':
        from .asset_compat import rebuild_a1
        cfg.scene.robot.spawn.usd_path = rebuild_a1(SOURCE, Path(output))
    if os.environ.get('WORLD_A1_FRICTION_COMPAT', '1') == '1' and RUNTIME == '6.0.1.0':
        from .friction_compat import articulation_class
        cfg.scene.robot.class_type = articulation_class(output)
    return cfg


def metrics(env):
    robot=env.scene['robot'];ids,names=robot.find_bodies('R.*_foot')
    return {'success':None,'success_definition':'Native handstand reward components and illegal-contact termination; no binary SR',
        'raised_foot_names':names,'rear_foot_heights_m':robot.data.body_pos_w[0,ids,2].detach().cpu().tolist(),
        'target_raised_foot_height_m':.5,'projected_gravity_b':robot.data.projected_gravity_b[0].detach().cpu().tolist(),
        'target_gravity_b':[1.,0.,0.],
        'native_horizon_seconds':10.,'push_interval_seconds':[10.,15.],
        'push_coverage_note':'Native episode is10s; a10-15s interval push is generally not reached. No injected earlier push.'}


def instruction(task_id,env=None):
    return '''Perform the original Unitree A1 flat handstand task: raise REAR legs and support the body on FRONT feet. Native handstand_type="back" means back legs in air, not standing on back legs. Native raised rear-foot height target is0.5m and projected-gravity target in the body frame is[1,0,0]. Simultaneously follow the native sampled base velocity commands as specified by the upstream reward; commands were not replaced by zero.
Action is12 joint-position residuals, target=default_joint_position+0.25*u rad, explicit order[FR_hip,FR_thigh,FR_calf,FL_hip,FL_thigh,FL_calf,RR_hip,RR_thigh,RR_calf,RL_hip,RL_thigh,RL_calf], all names end _joint. Native config clips processed joint-position targets to[-100,100]rad, not a promise of physical safety and not an input±1limit. Native motor PD tracks joint targets; no handstand policy, gait or whole-body stabilizer is supplied. Use runtime scale/offset/joint limit metadata.
Only native noisy actor terms are provided: body angular velocity(3,scale.25), projected gravity(3), velocity_commands(3), joint positions relative to defaults(12,scale1), joint velocities(12,scale.05), previous actions(12). This is45values, normally delivered as named native_policy_terms. Base linear velocity and height scans are absent from actor; no critic data is exposed.
One control tick=.02s; four .005s physical substeps. The FIXED SOURCE native horizon is10s=500steps (the summary material incorrectly says20s). Preserve it. The inherited random velocity-push interval10-15s usually does not occur before this native timeout; do not claim actual push recovery without evidence. We do not move the push earlier or extend horizon to manufacture coverage.
Native reward emphasizes rear foot height, feet-air/contact events, handstand gravity orientation, command tracking, torque/joint limits and smoothness. Non-foot illegal contact can terminate the task. There is no official binary full-episode handstand success rate; evaluator records native weighted components and terminations and separately reports rear-foot heights/gravity. Surviving, saying completed, or reaching a transient pose is not automatically success.
For fast balance feedback you may use coding_control if available: code reads only current actor terms and returns this same12D native action each tick. Long open-loop joint holds can tip the robot. Keep working until native termination or the step budget, without accessing simulator files or privileged judge data.'''


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
            texture = Path(__file__).resolve().parents[3] / 'third_party/benchmarks/robot_lab/assets/Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr'
            if not texture.is_file():raise FileNotFoundError(texture)
            value.texture_file = str(texture)
        children = value.values() if isinstance(value, dict) else value if isinstance(value, (list, tuple)) else vars(value).values() if hasattr(value, '__dict__') else ()
        for child in children:visit(child)
    visit(cfg)
