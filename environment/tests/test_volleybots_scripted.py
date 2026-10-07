import math
from environment.benchmarks.volleybots.scripted_opponent import ScriptedOpponent, manifest
from environment.benchmarks.volleybots.duel_isaac6 import instructions


def test_hover_rotors_and_saturation():
    obs=[0.]*37
    obs[2]=2.;obs[3]=1.;obs[18]=1.
    action=ScriptedOpponent.rotors(obs,[0.,0.,2.],[0.,0.,0.])
    assert len(action)==4 and all(-1<x<1 for x in action)
    thrust=sum((x+1)*6.003/2 for x in action)
    assert math.isclose(thrust,1.52*9.81,rel_tol=1e-6)
    assert all(-1<=x<=1 for x in ScriptedOpponent.rotors(obs,[1e3,1e3,1e3],[0.,0.,0.]))


def test_variant_disclosure():
    m=manifest()
    assert len(m['sha256'])==64 and not m['official_pretrained_baseline']
    text=instructions('T05')
    assert 'World-authored' in text and '37D' in text


def test_rotor_law_respects_native_half_turn_symmetry():
    # Rotate both state and target 180deg about Z: body rotor commands must agree.
    a=[0.]*37;a[2]=2.;a[3]=1.;a[18]=1.
    b=a.copy();b[3]=0.;b[6]=1.
    first=ScriptedOpponent.rotors(a,[.1,.05,2.],[0.,0.,0.])
    rotated=ScriptedOpponent.rotors(b,[-.1,-.05,2.],[0.,0.,0.])
    assert all(math.isclose(x,y,abs_tol=1e-12) for x,y in zip(first,rotated))


def test_fixed_opponent_protocol_is_pinned():
    import hashlib,json
    from pathlib import Path
    from environment.benchmarks.volleybots.fixed_rally_opponent import manifest as fixed_manifest
    from environment.evaluation.rollout_catalog import profile
    root=Path(__file__).resolve().parents[2]
    spec=json.loads((root/'third_party/benchmarks/volleybots/fixed-rally-protocol.json').read_text())
    for name,digest in spec['source_hashes'].items():
        assert hashlib.sha256((root/'environment/benchmarks/volleybots'/name).read_bytes()).hexdigest()==digest
    assert fixed_manifest()['version']==spec['opponent']
    assert profile('volleybots','T05')==spec['runtime_profile']
    assert spec['step_limit']==1000 and spec['draw_score']==0


def test_player_prompt_has_correct_win_and_coordinates(monkeypatch):
    monkeypatch.setenv('WORLD_VOLLEY_CODING_CONTROL','1')
    for player in (0,1):
        prompt=instructions('T05',player)
        assert f'Your native player index is {player}' in prompt
        assert f'Play native VolleyBots1v1 as drone{player}' in prompt
        assert 'positive X is your own half' in prompt
        assert "obs['native_policy_observation']" in prompt
        assert 'rotor yaw reaction signs are [-1,-1,+1,+1]' in prompt


def test_default_match_is_codex_receiving(monkeypatch):
    monkeypatch.delenv('WORLD_VOLLEY_PLAYER',raising=False)
    prompt=instructions('T05')
    assert 'Your native player index is 1' in prompt
    assert 'Play native VolleyBots1v1 as drone1' in prompt
    assert 'Native player0 serves initially' in prompt
    assert 'positive X is your own half' in prompt


def test_disabled_coding_prompt(monkeypatch):
    monkeypatch.setenv('WORLD_VOLLEY_CODING_CONTROL','0')
    prompt=instructions('T05')
    assert 'Coding control is disabled' in prompt
    assert 'You may write your own attitude/position' not in prompt
    assert 'coding_control can implement' not in prompt


def test_opponent_hashes_checked_before_simulation(tmp_path,monkeypatch):
    import json
    import pytest
    from environment.benchmarks.volleybots import duel_isaac6 as duel
    assert duel.verify_opponent()['version']=='fixed-rally-v1'
    monkeypatch.setattr(duel,'SOURCE',tmp_path/'checkout')
    (tmp_path/'fixed-rally-protocol.json').write_text(json.dumps({
        'source_hashes':{'fixed_rally_opponent.py':'0'*64},'opponent':'fixed-rally-v1'}))
    with pytest.raises(RuntimeError,match='source hash mismatch'):
        duel.verify_opponent()


def test_scripted_profile_does_not_replace_solo_or_native_contract():
    from environment.runtime.native_project_launch import load_project
    from environment.evaluation.rollout_catalog import profile
    _,base=load_project('volleybots')
    _,duel=load_project('volleybots','isaac6-scripted-1v1')
    assert base['commit']==duel['commit']
    assert base['tasks']==duel['tasks']
    assert profile('volleybots','T05-single')=='isaac6'
    assert base['adapter']=='environment.benchmarks.volleybots.project'
    assert duel['adapter']=='environment.benchmarks.volleybots.duel_isaac6'


def test_launcher_forwards_player_and_coding(tmp_path,monkeypatch,capsys):
    import json,sys
    from environment.runtime import native_project_launch as launch
    monkeypatch.setattr(launch,'verify_source',lambda *args:'test-only')
    monkeypatch.delenv('WORLD_AGENT_BACKEND',raising=False)
    monkeypatch.setenv('WORLD_VOLLEY_PLAYER','1')
    output=tmp_path/'unused'
    monkeypatch.setattr(sys,'argv',['launch','--project','volleybots','--task','T05',
        '--runtime-profile','isaac6-scripted-1v1','--mode','zero','--output',str(output),'--dry-run'])
    launch.main()
    command=json.loads(capsys.readouterr().out)
    assert 'WORLD_VOLLEY_PLAYER=1' in command
    assert 'WORLD_VOLLEY_CODING_CONTROL=0' in command
    assert 'world/volleybots:isaac6.0.1-experimental' in command
    assert not output.exists()
