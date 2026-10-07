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
        spec=importlib.util.spec_from_file_location('world_wheeled_quadruped_isaac601',compat)
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
    (OUTPUT/'wheeled_quadruped-compatibility.json').write_text(json.dumps({'runtime':runtime,'experimental':runtime=='6.0.1.0','upstream_files_modified':False,'play_config_used':False,'actor_group':'policy only; no critic'},indent=2))

def _namespace(name,path):
    if name not in sys.modules:
        module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module

def _packages():
    root=SOURCE/'source/wheeled_quadruped/wheeled_quadruped'
    for suffix in ['', '.tasks', '.tasks.balance']:
        _namespace('wheeled_quadruped'+suffix,root.joinpath(*suffix.strip('.').split('.')) if suffix else root)


def build(task_id, seed, output):
    if task_id not in ['T09','Wheeled-Quadruped-Balance-v0']:raise ValueError(task_id)
    module=importlib.import_module('wheeled_quadruped.tasks.balance.balance_env_cfg')
    cfg=module.WheeledQuadrupedBalanceEnvCfg()
    cfg.scene.num_envs=1;cfg.seed=seed;cfg.wait_for_textures=False
    _localize_debug_markers(cfg)
    return cfg


def metrics(env):
    return {'success':None,'success_definition':'Native rewards/fall terminations only; no published binary success rate',
            'final_base_height_m':float(env.scene['robot'].data.root_pos_w[0,2].item()),
            'target_height_m':.828}


def instruction(task_id,env=None):
    return '''Balance the native wheeled quadruped upright on its two REAR wheels, at its native target base height0.828m. This is not ordinary four-wheel driving. Front wheels are fixed; only two front-thigh joints and two rear-wheel joints are actuated. No learned balancing policy is supplied.
The native action concatenates thigh_pos then wheel_vel: front left thigh position residual, front right thigh position residual, rear left wheel velocity, rear right wheel velocity. Positions=default joint position+0.5*u radians; wheel velocity=5*u rad/s. Native action terms do not define an explicit clipping interval: consult runtime metadata and physical joint limits; do not assume [-1,1] is a native hard limit. Joint order resolved at runtime is authoritative. Only motor-level PD/velocity tracking is provided; you must produce balance control.
Actor observation order: body angular velocity(3), projected gravity(3), front-thigh positions relative to default(2), all four relative joint velocities(4 in runtime articulation order), previous actions(4). Native uniform noise remains enabled. No base linear velocity or base height from the privileged critic is provided. Do not infer hidden judge access from this task description.
One action is0.02s, 4physics substeps of0.005s, native horizon20s=1000steps. Maintain posture and reject horizontal velocity pushes sampled in x/y from[-0.5,0.5]m/s every10-15s. Native random mass, friction, reset state and observation noise remain active. Tilt beyond pi/3 or base height below0.4m ends the episode. These are native fall criteria, not proof of success; the evaluator reports native height/orientation/alive/effort/smoothness rewards and terminations, without invented binary SR.
If coding_control is available, write a short feedback controller using the supplied current actor vector on every control tick; no files, privileged simulator state or pretrained policy. Simulation advances only through authorized action tools. Long constant wheel commands may overturn the robot. Read runtime observation/action metadata before indexing vectors.'''


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
