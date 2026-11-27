"""Negative evidence tests; these fixtures are not physical success demonstrations."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('close_drawer_grader', ROOT / 'tasks/close-drawer/tests/grade.py')
grade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grade)
# Patch the shared implementation globals when exercising optional artifact QA.
grade = grade._grader


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    original_validate = grade.audit_artifacts
    monkeypatch.setattr(grade, 'audit_artifacts', lambda runs, expected=None: original_validate(runs, expected if expected is not None else {'benchmark': 'robocasa', 'task': 'CloseDrawer', 'horizon': 1, 'action_dimension': 12}))
    (tmp_path / 'finalized').write_text('complete\n')
    result = dict(protocol='robotworld.episode.v1', episode_id='e', control_step=1,
                  observation_id='e:1', horizon=1, terminal=True, termination='budget_exhausted', success=False)
    write(tmp_path / 'episode.json', result)
    write(tmp_path / 'evaluation-config.json', dict(benchmark='robocasa', task='CloseDrawer', native_horizon=1, control_frequency_hz=20))
    write(tmp_path / 'native-steps.jsonl', dict(control_step=1, action=[0]*12, success=False, terminated=False, truncated=False))
    request = dict(episode_id='e', observation_id='e:0', request_id='r', commands=[{'arguments': {'steps': 1}}])
    receipt = dict(result, request_id='r', executed_steps=1, executed_steps_per_command=[1], error=None, execution_uncertain=False)
    (tmp_path / 'events.jsonl').write_text('\n'.join(json.dumps(x) for x in [dict(event='execution_requested', data=request), dict(event='execution_completed', data=receipt)]))
    (tmp_path / 'video.mp4').touch()
    monkeypatch.setattr(grade, 'verify_video', lambda *args: {'decoded': True, 'frames':2})
    return tmp_path


def test_valid_budget_failure(evidence):
    result = grade.audit_artifacts(evidence)
    assert result['status'] == 'valid_failure'
    assert result['reward'] == 0


def test_missing_native_trace_only_valid_for_zero_step_episode(evidence):
    (evidence / 'native-steps.jsonl').unlink()
    with pytest.raises(FileNotFoundError):
        grade.audit_artifacts(evidence)
    episode = json.loads((evidence / 'episode.json').read_text())
    episode.update(control_step=0, observation_id='e:0', termination='agent_stopped')
    write(evidence / 'episode.json', episode)
    (evidence / 'events.jsonl').write_text('')
    assert grade.audit_artifacts(evidence)['episode']['control_step'] == 0


def test_missing_finalization(evidence):
    (evidence / 'finalized').unlink()
    with pytest.raises(FileNotFoundError):
        grade.audit_artifacts(evidence)


@pytest.mark.parametrize('change', [dict(success=True), dict(termination='success'), dict(termination='infrastructure_error'), dict(control_step=2), dict(horizon=2)])
def test_contradictory_result(evidence, change):
    path = evidence / 'episode.json'
    result = json.loads(path.read_text())
    result.update(change)
    write(path, result)
    with pytest.raises(grade.InvalidTrial):
        grade.audit_artifacts(evidence)


def test_native_steps_must_be_contiguous(evidence):
    path = evidence / 'native-steps.jsonl'
    row = json.loads(path.read_text())
    row['control_step'] = 2
    write(path, row)
    with pytest.raises(grade.InvalidTrial, match='Non-contiguous'):
        grade.audit_artifacts(evidence)


def test_pending_request_invalid(evidence):
    path = evidence / 'events.jsonl'
    path.write_text(path.read_text().splitlines()[0])
    with pytest.raises(grade.InvalidTrial, match='Incomplete'):
        grade.audit_artifacts(evidence)


def test_feedback_control_journal(evidence):
    from environment.service.episode import Episode
    from environment.service.feedback import FeedbackAdapter

    events = []
    def journal(kind, data):
        events.append(dict(event=kind, data=data))

    adapter = SimpleNamespace(describe=lambda: {'tools': []}, observe=lambda: {'state': {}},
                              prepare=lambda commands: [(commands[0], 1)],
                              step=lambda action: dict(success=False, terminated=False, truncated=False))
    episode = Episode(FeedbackAdapter(adapter, journal), 1, 'e', journal=journal)
    command = {'tool': 'coding_control', 'arguments': {'note': '', 'max_steps': 5,
               'code': 'def control(obs, memory): return {"tool":"move","arguments":{"steps":1}}'}}
    receipt = episode.execute(dict(episode_id='e', observation_id='e:0', request_id='r', commands=[command]))
    assert receipt['executed_steps'] == 1
    assert [event['event'] for event in events] == ['execution_requested', 'feedback_callback', 'execution_completed']
    path = evidence / 'events.jsonl'
    path.write_text('\n'.join(map(json.dumps, events)))
    assert grade.audit_artifacts(evidence)['status'] == 'valid_failure'

    write(evidence / 'execution-config.json', {'code_control': True})
    duplicate = {**events[1], 'data': {**events[1]['data'], 'program_step': 1}}
    for journal in ([events[0], events[-1]], [events[0], events[1], events[1], events[-1]],
                    [events[0], events[1], duplicate, events[-1]]):
        path.write_text('\n'.join(map(json.dumps, journal)))
        with pytest.raises(grade.InvalidTrial, match='Feedback callback count'):
            grade.audit_artifacts(evidence)

    command['arguments']['max_steps'] = 0
    path.write_text('\n'.join(map(json.dumps, events)))
    with pytest.raises(grade.InvalidTrial, match='Invalid executed count'):
        grade.audit_artifacts(evidence)


def test_callbacks_follow_real_multi_program_receipt(evidence):
    from environment.service.episode import Episode
    from environment.service.feedback import FeedbackAdapter

    events = []
    def journal(kind, data):
        events.append(dict(event=kind, data=data))

    native = SimpleNamespace(describe=lambda: {'tools': []}, observe=lambda: {},
                             prepare=lambda commands: [(commands[0], 1)],
                             step=lambda action: dict(success=False, terminated=False, truncated=False))
    episode = Episode(FeedbackAdapter(native, journal), 1, 'e', journal=journal)
    bodies = ['return missing_variable', 'return {"bad": 1}', 'return {"done": True}',
              'return {"tool": "move", "arguments": {"steps": 1}}', 'return {"done": True}']
    commands = [{'tool': 'coding_control', 'arguments': {'max_steps': 5, 'note': '',
                 'code': 'def control(obs, memory): ' + body}} for body in bodies]
    receipt = episode.execute(dict(episode_id='e', observation_id='e:0', request_id='r', commands=commands))
    assert receipt['executed_steps_per_command'] == [0, 0, 0, 1, 0]
    assert [f['command_index'] for f in receipt['feedback']] == [0, 1]
    path = evidence / 'events.jsonl'
    path.write_text('\n'.join(map(json.dumps, events)))
    assert grade.audit_artifacts(evidence)['status'] == 'valid_failure'

    write(evidence / 'execution-config.json', {'code_control': False})
    with pytest.raises(grade.InvalidTrial, match='Feedback control disabled'):
        grade.audit_artifacts(evidence)


def test_feedback_done_and_unreached_program_in_mixed_request(evidence):
    rows = grade.read_jsonl(evidence / 'events.jsonl')
    program = {'tool': 'coding_control', 'arguments': {'max_steps': 5}}
    rows[0]['data']['commands'] = [program, {'tool': 'move', 'arguments': {'steps': 1}}, program]
    rows[-1]['data'].update(executed_steps_per_command=[0, 1, 0], feedback=[])
    rows.insert(1, {'event': 'feedback_callback', 'data': {'program_step': 0, 'reply': {'action': {'done': True}}}})
    (evidence / 'events.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    write(evidence / 'execution-config.json', {'code_control': True})
    assert grade.audit_artifacts(evidence)['status'] == 'valid_failure'
    (evidence / 'events.jsonl').write_text('\n'.join(map(json.dumps, [rows[0], rows[-1]])))
    with pytest.raises(grade.InvalidTrial, match='Feedback callback count'):
        grade.audit_artifacts(evidence)


def test_planner_callback_can_cover_multiple_physical_steps(evidence):
    expected = {'benchmark': 'robodojo', 'task': 'HangMugs', 'horizon': 4}
    write(evidence / 'evaluation-config.json', {**expected, 'native_horizon': 4})
    write(evidence / 'execution-config.json', {'code_control': True})
    path = evidence / 'episode.json'
    result = json.loads(path.read_text())
    result.update(control_step=4, horizon=4, observation_id='e:4')
    write(path, result)
    write(evidence / 'result.json', {'success': False})
    write(evidence / 'native-steps.jsonl', {'control_step_before': 0, 'control_step': 4, 'executed_steps': 4,
                                             'success': False, 'terminated': False, 'truncated': False})
    rows = grade.read_jsonl(evidence / 'events.jsonl')
    rows[0]['data']['commands'] = [{'tool': 'coding_control', 'arguments': {'max_steps': 1}}]
    rows[-1]['data'].update(result, executed_steps_per_command=[4], executed_steps=4)
    rows[-1]['data']['feedback'] = [{'command_index': 0, 'details': {'motion_accepted': True}}]
    rows.insert(1, {'event': 'feedback_callback', 'data': {'program_step': 0}})
    (evidence / 'events.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    assert grade.audit_artifacts(evidence, expected)['status'] == 'valid_failure'
    del rows[-1]['data']['feedback']
    (evidence / 'events.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    assert grade.audit_artifacts(evidence, expected)['status'] == 'valid_failure'


def test_planner_terminal_after_zero_step_motion_at_same_counter(evidence):
    from environment.service.episode import Episode
    from environment.service.feedback import FeedbackAdapter

    events, native = [], []
    def journal(kind, data):
        events.append(dict(event=kind, data=data))

    def motion(action):
        success = bool(native)
        result = dict(success=success, terminated=success, truncated=False, executed_steps=0,
                      details={'motion_accepted': False})
        native.append(dict(control_step_before=0, control_step=0, **result))
        return result

    adapter = SimpleNamespace(describe=lambda: {'tools': []}, observe=lambda: {},
                              prepare=lambda commands: [(commands[0], 1)], step=motion)
    episode = Episode(FeedbackAdapter(adapter, journal), 4, 'e', journal=journal)
    command = {'tool': 'coding_control', 'arguments': {'note': '', 'max_steps': 1,
               'code': 'def control(obs, memory): return {"tool":"move_eef","arguments":{}}'}}
    for request_id in ['first', 'second']:
        episode.execute(dict(episode_id='e', observation_id='e:0', request_id=request_id, commands=[command]))
    assert episode.status()['success'] and episode.status()['control_step'] == 0
    write(evidence / 'episode.json', episode.status())
    write(evidence / 'result.json', {'success': True})
    (evidence / 'native-steps.jsonl').write_text('\n'.join(map(json.dumps, native)))
    (evidence / 'events.jsonl').write_text('\n'.join(map(json.dumps, events)))
    expected = {'benchmark': 'robodojo', 'task': 'HangMugs', 'horizon': 4}
    write(evidence / 'evaluation-config.json', {**expected, 'native_horizon': 4})
    assert grade.audit_artifacts(evidence, expected)['reward'] == 1


def test_feedback_callback_needs_pending_program(evidence):
    path = evidence / 'events.jsonl'
    rows = grade.read_jsonl(path)
    rows.insert(1, dict(event='feedback_callback', data={}))
    path.write_text('\n'.join(map(json.dumps, rows)))
    with pytest.raises(grade.InvalidTrial, match='Feedback callback outside'):
        grade.audit_artifacts(evidence)


def test_uncertain_receipt_invalid(evidence):
    path = evidence / 'events.jsonl'
    rows = grade.read_jsonl(path)
    rows[-1]['data']['execution_uncertain'] = True
    path.write_text('\n'.join(map(json.dumps, rows)))
    with pytest.raises(grade.InvalidTrial, match='Uncertain'):
        grade.audit_artifacts(evidence)


def test_success_requires_native_confirmation(evidence):
    for filename in ['episode.json', 'native-steps.jsonl']:
        path = evidence / filename
        row = json.loads(path.read_text())
        row['success'] = True
        if filename == 'episode.json':
            row['termination'] = 'success'
        write(path, row)
    path = evidence / 'events.jsonl'
    rows = grade.read_jsonl(path)
    rows[-1]['data'].update(success=True, termination='success')
    path.write_text('\n'.join(map(json.dumps, rows)))
    assert grade.audit_artifacts(evidence)['reward'] == 1


def test_invalid_does_not_write_failure_reward(evidence, monkeypatch):
    output = evidence / 'verifier'
    output.mkdir()
    (output / 'reward.txt').write_text('0\n')
    outcome = json.loads((evidence / 'episode.json').read_text())
    outcome['termination'] = 'infrastructure_error'
    write(evidence / 'episode.json', outcome)
    monkeypatch.setattr('sys.argv', ['grade.py', '--runs', str(evidence), '--output', str(output)])
    with pytest.raises(grade.InvalidTrial):
        grade.main()
    assert not (output / 'reward.txt').exists()
    assert json.loads((output / 'details.json').read_text())['status'] == 'invalid'


def test_invalid_video_rejected(tmp_path):
    path = tmp_path / 'video.mp4'
    path.write_bytes(b'not a video')
    with pytest.raises(Exception):
        grade.verify_video(path, 1, 20)


def test_other_task_requires_authoritative_result(evidence):
    path=evidence/'evaluation-config.json'
    config=json.loads(path.read_text());config.update(benchmark='ai_cps',task='22');write(path,config)
    expected={'benchmark':'ai_cps','task':'22','horizon':1}
    with pytest.raises(grade.InvalidTrial,match='Missing authoritative'):
        grade.audit_artifacts(evidence,expected)
    write(evidence/'result.json',{'success':None,'status':'incomplete_diagnostic'})
    with pytest.raises(grade.InvalidTrial,match='cannot invent'):
        grade.audit_artifacts(evidence,expected)


def test_early_return_is_incomplete_without_inventing_native_success(evidence):
    path=evidence/'evaluation-config.json'
    config=json.loads(path.read_text());config.update(benchmark='ai_cps',task='22',native_horizon=300);write(path,config)
    path=evidence/'episode.json';result=json.loads(path.read_text());result.update(horizon=300,termination='agent_stopped');write(path,result)
    path=evidence/'events.jsonl';rows=grade.read_jsonl(path);rows[-1]['data'].update(horizon=300,termination=None,terminal=False);path.write_text('\n'.join(map(json.dumps,rows)))
    write(evidence/'result.json',{'success':None,'status':'incomplete_diagnostic'})
    report=grade.audit_artifacts(evidence,{'benchmark':'ai_cps','task':'22','horizon':300})
    assert report['status']=='valid_failure' and report['reward'] == 0
    assert report['native_success'] is None
    assert report['score_basis'] == 'goal_not_confirmed_before_agent_return'


def test_wrong_expected_task_rejected(evidence):
    with pytest.raises(grade.InvalidTrial,match='Wrong task'):
        grade.audit_artifacts(evidence,{'benchmark':'robocasa','task':'OtherTask'})


def test_service_identity_takes_precedence_over_legacy_configuration(evidence):
    config=json.loads((evidence/'evaluation-config.json').read_text())
    write(evidence/'service-config.json',config)
    write(evidence/'evaluation-config.json',{'legacy_launch_args':{}})
    assert grade.audit_artifacts(evidence)['status']=='valid_failure'


def capped_evidence(evidence):
    path=evidence/'evaluation-config.json';config=json.loads(path.read_text());config['native_horizon']=10;write(path,config)
    fields=dict(horizon=10,termination='agent_stopped',control_step_limit=1,
                effective_control_step_limit=1,control_step_limit_overshoot=0,stop_reason='control_step_limit')
    path=evidence/'episode.json';result=json.loads(path.read_text());result.update(fields);write(path,result)
    path=evidence/'events.jsonl';rows=grade.read_jsonl(path);rows[-1]['data'].update(fields);path.write_text('\n'.join(map(json.dumps,rows)))
    write(evidence/'execution-config.json',dict(episode_id='e',native_horizon=10,control_step_limit=1))
    return dict(benchmark='robocasa',task='CloseDrawer',horizon=10,action_dimension=12)


def test_configured_cap_preserves_original_verifier_horizon(evidence):
    report=grade.audit_artifacts(evidence,capped_evidence(evidence))
    assert report['reward']==0
    assert report['score_basis']=='goal_not_confirmed_before_control_step_limit'
    assert report['episode']['horizon']==10


def test_cap_needs_simulator_owned_configuration(evidence):
    expected=capped_evidence(evidence)
    (evidence/'execution-config.json').unlink()
    with pytest.raises(grade.InvalidTrial,match='Unconfigured cap'):
        grade.audit_artifacts(evidence,expected)


def test_early_receipt_cannot_fabricate_cap_terminal(evidence):
    expected=capped_evidence(evidence)
    path=evidence/'execution-config.json';config=json.loads(path.read_text());config['control_step_limit']=2;write(path,config)
    for path in [evidence/'episode.json',evidence/'events.jsonl']:
        path.write_text(path.read_text().replace('"control_step_limit": 1','"control_step_limit": 2').replace('"effective_control_step_limit": 1','"effective_control_step_limit": 2'))
    with pytest.raises(grade.InvalidTrial,match='Cap stop reason contradicts terminal state'):
        grade.audit_artifacts(evidence,expected)
