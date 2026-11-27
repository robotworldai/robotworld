import json
import tomllib
from pathlib import Path
import pytest
import yaml
from environment.harbor_tasks.export import inventory, select, export_task, ROOT, native_service
from environment.service.native_project import NativeProjectAdapter


def test_inventory_does_not_pretend_unported_tasks_are_ready():
    rows = inventory()
    assert len(rows) == 84
    assert len({r['benchmark'] for r in rows}) == 20
    counter = next(r for r in rows if r['task'] == 'CountertopCleanup')
    assert counter['harbor_status'] == 'adapter_ready'
    assert counter['native_steps'] is None and counter['horizon'] == 600
    assert 'no verified official horizon' in counter['budget_override_reason']



def test_export_preserves_native_budget_and_selected_task(tmp_path):
    row = select(inventory(), ['robocasa'], ['CoffeeSetupMug'])[0]
    output = tmp_path/'coffee'
    export_task(row, output, ROOT/'tasks/close-drawer', seed=19)
    cfg = tomllib.loads((output/'task.toml').read_text())
    assert cfg['metadata']['name'] == 'coffee-setup-mug'
    compose = yaml.safe_load((output/'environment/docker-compose.yaml').read_text())
    command = compose['services']['simulator']['command']
    assert command[command.index('--task') + 1] == 'CoffeeSetupMug'
    assert command[command.index('--seed') + 1] == '19'
    assert json.loads((output/'tests/task.json').read_text())['horizon'] == 600
    assert '600 native control steps' in (output/'instruction.md').read_text()


def test_native_export_selects_world_scoring_without_changing_native_budget():
    row = select(inventory(), ['omniisaacgymenvs'])[0]
    cfg = native_service(row, 7, {})
    assert row['native_steps'] == 1000
    assert row['scoring_profile'] == 'world-state-v1'
    assert cfg['command'][cfg['command'].index('--steps')+1] == str(row['horizon'])
    assert 'codex' not in ' '.join(cfg['command'])
    assert cfg['environment']['WORLD_EPISODE_TOKEN'].startswith('${')
    assert cfg['environment']['WORLD_CONTROL_STEP_LIMIT'] == '${WORLD_CONTROL_STEP_LIMIT:-}'


class FakeSimulator:
    steps = 0
    dt = .02
    policy_images = {}
    action_metadata = {'dim':2, 'lower':[-1,-1], 'upper':[1,1], 'terms':[]}
    done = False
    truncated = False
    def observation(self): return {'x':self.steps}
    def step(self, action): self.steps += 1
    def result(self): return {'success':self.steps >= 2, 'privileged_evaluator_value':42}
    def close(self): self.closed = True


def test_native_service_preserves_actor_boundary_and_success(tmp_path):
    sim = FakeSimulator()
    adapter = NativeProjectAdapter(sim,'digit','T13','instruction',700,tmp_path)
    assert 'privileged_evaluator_value' not in json.dumps(adapter.observe())
    cmds = [{'tool':'apply_action','arguments':{'action':[0.,1.], 'steps':2,'note':''}}]
    assert adapter.prepare(cmds) == [([0.,1.],2)]
    assert not adapter.step([0.,1.])['success']
    assert adapter.step([0.,1.])['success']
    assert len(adapter.history) == 3
    adapter.observe(); adapter.observe()
    assert len(adapter.history) == 3
    adapter.close(); adapter.close()
    assert sim.closed
    assert len((tmp_path/'native-steps.jsonl').read_text().splitlines()) == 2


@pytest.mark.parametrize('stop_reason,native_complete,truncated,expected', [
    ('world_evaluation_end', False, False, 'budget_exhausted'),
    ('native_termination', True, False, 'native_failure'),
    ('native_termination', True, True, 'native_truncation'),
    ('world_scene_failure', False, False, 'native_failure'),
])
@pytest.mark.parametrize('flag_source', ['simulator', 'evaluation'])
def test_native_service_terminal_reason_at_horizon(tmp_path, monkeypatch, stop_reason, native_complete,
                                                   truncated, expected, flag_source):
    from environment.service.episode import Episode

    sim = FakeSimulator()
    sim.done = True
    result = {'success': False, 'native_episode_complete': native_complete, 'stop_reason': stop_reason}
    if flag_source == 'simulator':
        sim.truncated = truncated
    else:
        # OmniDrones, ANYmal and VolleyBots publish the native flag in result().
        monkeypatch.delattr(FakeSimulator, 'truncated')
        result['evaluation'] = {'truncated': truncated}
    sim.result = lambda: result
    adapter = NativeProjectAdapter(sim, 'digit', 'T13', 'instruction', 1, tmp_path)
    episode = Episode(adapter, 1, 'test')
    command = {'tool':'apply_action', 'arguments':{'action':[0,0], 'steps':1, 'note':''}}
    receipt = episode.execute({'request_id':'one', 'episode_id':'test', 'observation_id':'test:0',
                               'commands':[command]})
    assert receipt['termination'] == expected
    trace = json.loads((tmp_path/'native-steps.jsonl').read_text())
    assert trace['terminated'] is (expected == 'native_failure')
    assert trace['truncated'] is (expected == 'native_truncation')


def test_native_chunk_prevalidation_does_not_step(tmp_path):
    sim = FakeSimulator()
    adapter = NativeProjectAdapter(sim,'digit','T13','instruction',700,tmp_path)
    with pytest.raises(ValueError):
        adapter.prepare([{'tool':'apply_action','arguments':{'action':[0,2], 'steps':1,'note':''}}])
    assert sim.steps == 0


def test_behavior_budget_cap_is_explicit_and_original_preserved():
    rows = select(inventory(), ['behavior_1k'])
    assert len(rows) == 10
    assert all(r['horizon'] == min(r['native_steps'],2000) for r in rows)
    assert all(r['native_steps'] > 2000 for r in rows)


def test_export_packages_mcp_client_for_builtin_agents(tmp_path):
    row = select(inventory(), ['robocasa'], ['CloseDrawer'])[0]
    output = tmp_path/'drawer'
    export_task(row, output, ROOT/'tasks/close-drawer')
    assert (output/'environment/client/environment/bridges/mcp.py').is_file()
    assert (output/'environment/client/environment/clients/episode.py').is_file()
    assert not (output/'environment/client/environment/service').exists()
    assert 'COPY client/' in (output/'environment/Dockerfile').read_text()


def test_numeric_task_ids_export_as_semantic_names_without_changing_identity():
    rows = inventory()
    ball = next(r for r in rows if r['benchmark']=='ai_cps' and r['task']=='22')
    piano = next(r for r in rows if r['benchmark']=='bench2dex' and r['task']=='41')
    assert ball['name']=='catch-moving-ball' and ball['task_key']=='22'
    assert piano['name']=='bimanual-piano-melody' and piano['task_key']=='41'


def test_task_collection_wait_is_distinct_from_verifier_budget(tmp_path):
    row = select(inventory(), ['robocasa'], ['CloseDrawer'])[0]
    output = tmp_path/'drawer'
    export_task(row, output, ROOT/'tasks/close-drawer')
    cfg = tomllib.loads((output/'task.toml').read_text())
    assert cfg['verifier']['timeout_sec'] == 60
    collect = cfg['verifier']['collect'][0]
    assert collect['timeout_sec'] == 310
    assert 'seq 1 300' in collect['command'] and 'sleep 1;' in collect['command']
    assert json.loads((output/'tests/task.json').read_text())['horizon'] == row['horizon']


def test_robolab_service_uses_unambiguous_service_output_flag():
    """Upstream's parse_known_args treats --output as an ambiguous abbreviation."""
    import argparse
    from environment.harbor_tasks.services import service
    rows = select(inventory(), ['robolab'])
    assert len(rows) == 10
    for row in rows:
        command = service(row, 7)['command']
        assert '--output' not in command
        assert command[command.index('--service-output') + 1] == '/runs'
        # Match the upstream pre-parser's two aliases; this fails for --output
        # even though it uses parse_known_args and the service has its own parser.
        parser = argparse.ArgumentParser()
        parser.add_argument('--output-folder-name', '--output_folder_name')
        parser.parse_known_args(command[2:])
        path = ROOT/'tasks'/row['name']/'environment/docker-compose.yaml'
        if path.exists():
            checked_in = yaml.safe_load(path.read_text())['services']['simulator']['command']
            assert '--output' not in checked_in
            assert '--service-output' in checked_in
