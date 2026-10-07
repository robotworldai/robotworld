import json
import sys
from types import SimpleNamespace

from environment.benchmarks.robodojo import diagnostics


def test_hold_result_explicitly_excludes_model_scoring(tmp_path,monkeypatch):
    monkeypatch.setattr(diagnostics,'scene_snapshot',lambda env:{})
    reward=SimpleNamespace(get_reward=lambda:False)
    env=SimpleNamespace(reward_manager=reward,get_obs=lambda:{'state':{}},
                        is_episode_end=lambda:True,take_action_cnt=[0],success=[False],step_lim=400)
    adapter=SimpleNamespace(observe_content=lambda:[],video=SimpleNamespace(close=lambda:None))
    diagnostics.run_hold_diagnostic(env,adapter,4,tmp_path)
    data=json.loads((tmp_path/'result.json').read_text())
    assert data['diagnostic_only'] is True and data['model_calls']==0


def test_snapshot_uses_simulation_stage_not_ui_context(monkeypatch):
    monkeypatch.setitem(sys.modules,'pxr',SimpleNamespace(Gf=None,UsdGeom=None))
    monkeypatch.setattr('environment.benchmarks.robodojo.compat.conveyor.snapshot',lambda:{})
    traversed=[]
    stage=SimpleNamespace(Traverse=lambda:traversed.append(True) or [])
    env=SimpleNamespace(sim=SimpleNamespace(sim=SimpleNamespace(stage=stage)),
                        scene_manager=SimpleNamespace(get_objects=lambda **kwargs:{}),
                        take_action_cnt=[4])
    result=diagnostics.scene_snapshot(env)
    assert traversed==[True] and result['env_step']==4
