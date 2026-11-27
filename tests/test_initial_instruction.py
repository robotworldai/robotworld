import json
import pytest
from environment.clients.episode import EpisodeClient
from environment.harbor_tasks.instructions import control_brief


def test_initial_instruction_uses_public_description_only(tmp_path):
    client = EpisodeClient('http://unused', 'secret-not-for-artifacts')
    calls = []
    description = {'instruction': 'Close the selected drawer',
                   'tools': [{'name': 'move_eef', 'description': 'base-relative delta'}],
                   'action_contract': 'One unit requests 0.05 m; not absolute position'}
    def request(path, body=None):
        calls.append((path, body))
        return description
    client.request = request
    text = client.prepare_instruction('Task brief', tmp_path)
    assert calls == [('describe', None)]
    assert description['action_contract'] in text
    assert json.loads((tmp_path/'initial-description.json').read_text()) == description
    assert (tmp_path/'effective-instruction.md').read_text() == text
    assert 'secret-not-for-artifacts' not in text


def test_incomplete_description_cannot_silently_start_agent(tmp_path):
    client = EpisodeClient('http://unused', 'private')
    client.request = lambda path: {'instruction': 'go'}
    with pytest.raises(ValueError):
        client.prepare_instruction('brief', tmp_path)
    assert not (tmp_path/'effective-instruction.md').exists()


def test_static_robocasa_contract_matches_runtime_source():
    from environment.benchmarks.action_contracts import action_contract
    assert action_contract('robocasa') in control_brief('robocasa')


def test_public_control_contract_is_also_in_system_prompt(tmp_path):
    client=EpisodeClient('http://unused','private-key')
    client.request=lambda path: {'tools':[{'name':'move'}],'action_contract':'absolute joint radians'}
    client.prepare_instruction('task',tmp_path)
    text=client.system_instruction('Control the robot.',tmp_path)
    assert 'absolute joint radians' in text and 'private-key' not in text
    assert (tmp_path/'effective-system-prompt.txt').read_text()==text


def test_prompt_deduplicates_tool_schemas_without_dropping_contract(tmp_path):
    from environment.clients.episode import prompt_description
    desc={'instruction':'goal','control_instructions':'robot geometry',
          'action_contract':'absolute joint radians', 'native_horizon':600,
          'tools':[{'name':'move','description':'unique-tool-description','parameters':{'type':'object'}}]}
    view=prompt_description(desc)
    assert view['available_robot_tools']==['move']
    assert view['control_instructions']=='robot geometry'
    assert view['action_contract']=='absolute joint radians'
    assert view['native_horizon']==600
    assert 'tools' not in view and desc['tools'][0]['description']=='unique-tool-description'


def test_every_native_task_has_a_legacy_robot_scene_brief():
    from environment.harbor_tasks.export import inventory, NATIVE_PROJECTS
    from environment.evaluation.task_names import routing_key
    from environment.benchmarks.operating_brief import operating_brief
    for row in inventory():
        if row['benchmark'] in NATIVE_PROJECTS:
            assert 'Robot and interaction geometry:' in operating_brief(
                'native_project',routing_key(row['benchmark'],row['task']))


def test_portable_clients_match_shared_prompt_builder():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    copies=list((root/'tasks').glob('*/environment/client/environment/clients/episode.py'))
    assert len(copies)==84
    for p in copies:assert p.read_bytes()==(root/'environment/clients/episode.py').read_bytes()


def test_generic_execute_keeps_nested_action_schemas(monkeypatch):
    from environment.clients.episode import prompt_description
    monkeypatch.setenv('WORLD_AGENT_INTERFACE','generic-v1')
    description={'tools':[{'name':'move_eef','description':'absolute pose','parameters':{'type':'object'}}]}
    assert prompt_description(description)==description
