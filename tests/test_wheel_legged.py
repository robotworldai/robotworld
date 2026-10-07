"""Checks that the adapter preserves native action semantics and task boundary."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from environment.benchmarks.wheel_legged import project

W = Path(__file__).resolve().parents[1]
P = W / 'third_party/benchmarks/wheel_legged'
TASKS = P / 'checkout/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot'


def test_pinned_real_assets_and_registered_tasks():
    spec = importlib.util.spec_from_file_location('wheel_legged_assets_check', P / 'prepare_assets.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.verify()
    assert len(manifest['bundled_asset_files']) == 12
    registration = (TASKS / '__init__.py').read_text()
    for task in json.loads((P/'project.json').read_text())['tasks'].values():
        assert task['id'] in registration and 'Play' not in task['id']
        assert task['steps'] == 2000


def test_prompt_matches_actual_vmc_config_not_default_action_class():
    tree = ast.parse((TASKS/'wheel_legged_flat_env_cfg.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VmcActionsCfg')
    call = next(n.value for n in cls.body if isinstance(n, ast.Assign))
    params = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    assert params['action_scale_theta'] == .35
    assert params['action_scale_l0'] == .06
    assert params['action_scale_vel'] == 24
    assert params['l0_offset'] == .237
    prompt = project.instruction('T07')
    assert '0.35*u' in prompt and '0.237+0.06*u' in prompt and '24*u' in prompt
    assert '36 values' in prompt and 'previous_action[30:36]' in prompt


def test_build_retains_native_config_and_routes_only_generated_assets(monkeypatch, tmp_path):
    sentinel = object()
    cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=4096, robot=SimpleNamespace(spawn=SimpleNamespace())),
                          actions=sentinel, rewards=sentinel, terminations=sentinel,
                          observations=sentinel, events=sentinel, curriculum=sentinel)
    monkeypatch.setattr(project.importlib, 'import_module', lambda _: SimpleNamespace(WheelLeggedRecoveryFlatEnvCfg=lambda:cfg))
    result = project.build('Wheel-Legged-Recovery-Flat-v0', 9, tmp_path)
    assert result.seed == 9 and result.scene.num_envs == 1
    assert result.scene.robot.spawn.usd_dir == str(tmp_path/'generated_assets/robot')
    assert result.world_task_id == 'T07'
    for name in ['actions','rewards','terminations','observations','events','curriculum']:
        assert getattr(result,name) is sentinel


def test_custom_native_step_preserved_and_reset_suppressed_only_after_arm():
    class Base:
        def __init__(self, cfg, render_mode): self.resets=0;self.outcomes=0;self.recorder_manager=SimpleNamespace(active_terms=[])
        def step(self, action): return ('original', action)
        def _reset_idx(self, ids): self.resets+=1
        def _record_recovery_outcomes(self, success, elapsed): self.outcomes+=1
    class Array:
        def __init__(self, values): self.values=values
        def tolist(self): return self.values
    env = project.make_env(SimpleNamespace(env_class=Base))
    assert env.step([1]) == ('original',[1])
    env._reset_idx([0]);assert env.resets == 1
    env._record_recovery_outcomes(Array([False]),Array([0.]))
    assert env.world_recovery_outcomes == []
    env.capture_terminal=True
    env._reset_idx([0]);assert env.resets == 1
    env._record_recovery_outcomes(Array([True,False]),Array([.55,2.5]))
    assert env.outcomes == 2
    assert env.world_recovery_outcomes == [{'success':True,'elapsed_seconds':.55},{'success':False,'elapsed_seconds':2.5}]


def test_reactive_no_claimed_binary_success():
    class Tensor:
        def __init__(self,v): self.v=v
        def sum(self):return self
        def any(self):return self
        def item(self):return self.v
    env=SimpleNamespace(world_recovery_outcomes=[{'success':True,'elapsed_seconds':.55}],
        _terrain_commanded_distance=Tensor(2.),_terrain_tracking_distance=Tensor(.8),
        _recovery_pending=Tensor(False),cfg=SimpleNamespace(world_task_id='T08'))
    result=project.metrics(env)
    assert result['success'] is None
    assert result['initial_recovery_success'] is None
    assert result['terrain_tracking_ratio'] == .4
    assert result['native_recovery_successes'] == 1


def test_copied_marker_config_localization_preserves_task_fields():
    marker = SimpleNamespace(usd_path='None/Isaac/Props/UIElements/arrow_x.usd',scale=(.5,.5,.5))
    command = SimpleNamespace(goal_vel_visualizer_cfg=SimpleNamespace(markers={'arrow':marker}))
    cfg = SimpleNamespace(commands={'native':command}, reward_weight=2.)
    cfg.cycle = cfg
    project._localize_debug_markers(cfg)
    assert Path(marker.usd_path).is_file()
    assert marker.scale == (.5,.5,.5) and cfg.reward_weight == 2.
