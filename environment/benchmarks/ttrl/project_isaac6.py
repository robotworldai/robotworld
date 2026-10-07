"""Explicit experimental Sim6 overlay; native TTRL actions and scoring are shared."""
from pathlib import Path
import json
import subprocess

from . import project as original
from . import compat

LAB_COMMIT = '21f7136325136ca3f6ca4e0a8125edffe5c24f7e'
LAB = original.ROOT / 'isaaclab21_checkout'


class Simulator(original.Simulator):
    def launch_app(self, headless):
        if not headless:
            raise ValueError('This isolated experimental profile is headless')
        actual = subprocess.check_output(['git', '-c', 'safe.directory='+str(LAB), '-C', str(LAB), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-c', 'safe.directory='+str(LAB), '-C', str(LAB), 'status', '--porcelain'], text=True)
        if actual != LAB_COMMIT or dirty or (LAB/'VERSION').read_text().strip() != '2.1.0':
            raise RuntimeError('Experimental TTRL must retain clean original pinned IsaacLab2.1.0')
        import isaacsim
        return compat.launch_app(str(Path(isaacsim.__file__).parent/'apps/isaacsim.exp.full.kit'))

    def prepare_runtime(self):
        compat.install(LAB)
        # Original runner initializes envs before task configs.
        import isaaclab.envs

    def camera_class(self):
        from .camera import Camera
        return Camera

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (self.output/'isaac6-runtime-boundary.json').write_text(json.dumps({
            'profile': 'isaac6', 'experimental': True, 'official_physics_equivalence_claimed': False,
            'isaaclab_commit': LAB_COMMIT, 'isaaclab_source': str(LAB),
            'native_task_config_changed': False, 'native_eval_overrides_shared_with_default': True,
            'native_serves_actions_actor_history_and_metrics_shared_with_default': True,
            'learned_predictor_enabled': False,
            'torch_compile_condition': 'TORCHDYNAMO_DISABLE=1: original Python/Torch functions execute eagerly; no reward formula replacement',
            'aerodynamic_force_frame_bridge': 'Original wrench buffers retain world-frame forces/torques; each physics write passes is_global=True to PhysX. Local-frame callers retain the original write path.',
            'review_camera': {'eye_offset': [-3.5, -3.6, 2.8], 'target_offset': [-.3, 0, .7], 'policy_exposed': False, 'physics_advance': False},
        }, indent=2))


def create_sim(task_id, seed, output, headless=True):
    return Simulator(task_id, seed, output, headless)
