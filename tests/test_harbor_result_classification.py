import pytest
from scripts.check_harbor_results import classify_trial


def evidence(reason='agent_stopped', success=False):
    return {'episode': {'termination': reason, 'success': success},
            'status': 'valid_success' if success else 'valid_failure',
            'reward': int(success), 'native_success': None}


def model():
    return {'agent': {'import_path': 'agents.robotworld:RobotWorldCodexAgent'}}


def test_normal_model_return_without_completion_is_failure_not_native_success():
    verified=evidence()
    result=classify_trial(model(), {}, verified)
    assert result['status']=='valid_failure' and result['reward']==0
    assert result['score_basis']=='goal_not_confirmed_before_agent_return'
    assert verified['native_success'] is None


@pytest.mark.parametrize('error',['AgentTimeoutError','RuntimeError','ProtocolError','RewardFileNotFoundError'])
def test_model_errors_are_invalid_even_after_native_success(error):
    with pytest.raises(ValueError, match=error):
        classify_trial(model(), {'exception_info': {'exception_type': error}}, evidence('success',True))


@pytest.mark.parametrize('name',['nop','agents.robotworld:RobotWorldProbeAgent','agents.robolab_probe:RoboLabProbeAgent'])
def test_probe_is_diagnostic_even_if_physically_successful(name):
    result=classify_trial({'agent':{'name':name}}, {}, evidence('success',True))
    assert result['status']=='diagnostic' and result['reward'] is None


def test_host_metadata_can_identify_external_probe():
    result=classify_trial({'agent':{'name':'external:Probe'}}, {'agent_result':{'metadata':{
        'classification':'infrastructure_probe_not_model_evaluation'}}}, evidence())
    assert result['status']=='diagnostic'


def test_missing_agent_identity_not_accepted():
    with pytest.raises(ValueError,match='Missing trusted'):
        classify_trial({}, {}, evidence())


def test_success_only_from_validated_simulator_evidence():
    assert classify_trial(model(), {}, evidence('success',True))['reward']==1


def test_legacy_no_reward_probe_is_diagnostic_only():
    assert classify_trial({'agent':{'name':'nop'}}, {'exception_info':{
        'exception_type':'RewardFileNotFoundError'}}, evidence())['status']=='diagnostic'


def test_reference_replay_never_counts_as_model_success():
    result=classify_trial({'agent':{'import_path':'agents.robotworld:RobotWorldReferenceReplayAgent'}}, {}, evidence('success',True))
    assert result['status']=='diagnostic' and result['reward'] is None


def test_missing_historical_step_count_is_unknown_not_mismatch():
    from agents.reference_replay import reference_step_count
    assert reference_step_count({'control_steps':None})==(None,'unavailable')
    assert reference_step_count({'control_steps':None,'interaction_budget':{'control_steps':380}})==(380,'result.interaction_budget.control_steps')


def test_simulator_step_limit_is_not_an_early_model_return():
    verified = evidence()
    verified['episode']['stop_reason'] = 'control_step_limit'
    result = classify_trial(model(), {}, verified)
    assert result['reward'] == 0
    assert result['score_basis'] == 'goal_not_confirmed_before_control_step_limit'


@pytest.mark.parametrize('plan', ['result.json', 'lock.json', 'malformed-lock', 'custom-agent'])
def test_missing_planned_trial_prevents_complete_score(tmp_path, monkeypatch, capsys, plan):
    import json
    import sys
    from scripts import check_harbor_results

    # A job can stop before Harbor creates the second trial's directory.
    trial = tmp_path / 'first'
    trial.mkdir()
    (trial / 'config.json').write_text(json.dumps(model()))
    planned = {'n_total_trials': 2} if plan == 'result.json' else {'trials': [{}, {}]}
    if plan == 'malformed-lock':
        (tmp_path / 'lock.json').write_text('{')
        (tmp_path / 'result.json').write_text(json.dumps({'n_total_trials': 2}))
    elif plan == 'custom-agent':
        trial_lock = {'task': {'name': 'close-drawer'}, **model()}
        (trial / 'lock.json').write_text(json.dumps(trial_lock))
        (tmp_path / 'lock.json').write_text(json.dumps({'trials': [
            trial_lock, {'task': {'name': 'hang-mugs'}, 'agent': {'import_path': 'external:Agent'}}]}))
    else:
        (tmp_path / plan).write_text(json.dumps(planned))
    monkeypatch.setattr(check_harbor_results, 'check_trial', lambda path: {
        'trial': str(path), 'status': 'valid_success', 'reward': 1,
        'benchmark': 'robocasa', 'task': 'CloseDrawer'})
    monkeypatch.setattr(sys, 'argv', ['check_harbor_results', str(tmp_path)])

    assert check_harbor_results.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report['total_trials'] == 2
    assert report['not_run_trials'] == 1
    assert report['valid_trials'] == 1
    assert report['task_macro_success_rate'] is None
    assert report['coverage'] is None


@pytest.mark.parametrize('result', ['{"exception_info":{"exception_type":"RuntimeError"}}', '{'])
def test_failed_probe_without_episode_is_excluded_from_evaluation(tmp_path, monkeypatch, capsys, result):
    import json
    import sys
    from scripts import check_harbor_results

    trial = tmp_path / 'failed-probe'
    trial.mkdir()
    (trial / 'config.json').write_text(json.dumps({'agent': {'name': 'nop'}}))
    (trial / 'result.json').write_text(result)
    monkeypatch.setattr(sys, 'argv', ['check_harbor_results', str(tmp_path)])
    assert check_harbor_results.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report['evaluation_trials'] == 0
    assert report['diagnostic_trials'] == 1
    assert report['invalid_trials'] == 1
    assert report['tasks'] == {}


@pytest.mark.parametrize('missing_model', [True, False])
def test_partial_job_counts_planned_model_and_probe_attempts(tmp_path, monkeypatch, capsys, missing_model):
    import json
    import sys
    from scripts import check_harbor_results

    model_lock = {'task': {'name': 'close-drawer'}, **model()}
    probe_lock = {'task': {'name': 'hang-mugs'}, 'agent': {'name': 'robotworld-smoke100'}}
    plan = [model_lock, probe_lock]
    if missing_model:
        plan.append(model_lock)
    (tmp_path / 'lock.json').write_text(json.dumps({'trials': plan}))
    trial = tmp_path / 'first'
    trial.mkdir()
    (trial / 'lock.json').write_text(json.dumps(model_lock))
    (trial / 'config.json').write_text(json.dumps(model()))
    monkeypatch.setattr(check_harbor_results, 'check_trial', lambda path: {
        'trial': str(path), 'status': 'valid_success', 'reward': 1,
        'benchmark': 'robocasa', 'task': 'CloseDrawer'})
    monkeypatch.setattr(sys, 'argv', ['check_harbor_results', str(tmp_path)])

    assert check_harbor_results.main() == int(missing_model)
    report = json.loads(capsys.readouterr().out)
    assert report['total_trials'] == len(plan)
    assert report['evaluation_trials'] == 1 + missing_model
    assert report['diagnostic_trials'] == 1
    assert report['missing_trial_directories'] == report['not_run_trials'] == len(plan) - 1
    assert report['coverage'] == report['tasks']['robocasa/CloseDrawer']['coverage'] == 1 / (1 + missing_model)
    assert report['task_macro_success_rate'] == (None if missing_model else 1)


@pytest.mark.parametrize('change', ['task', 'agent', 'model', 'digest', 'duplicate-id', 'extra'])
def test_wrong_trial_cannot_complete_job(tmp_path, monkeypatch, capsys, change):
    import copy
    import json
    import sys
    from scripts import check_harbor_results

    first = {'task': {'name': 'close-drawer', 'digest': 'sha256:aaa'},
             'agent': {'name': 'claude-code', 'model_name': 'first'}}
    second = copy.deepcopy(first)
    if change == 'task':
        second['task']['name'] = 'hang-mugs'
    elif change == 'agent':
        second['agent']['name'] = 'antigravity-cli'
    elif change == 'model':
        second['agent']['model_name'] = 'second'
    elif change == 'digest':
        second['task']['digest'] = 'sha256:bbb'

    planned = [first] if change == 'extra' else [first, second]
    (tmp_path / 'lock.json').write_text(json.dumps({'trials': planned}))
    for name in ('one', 'two'):
        trial = tmp_path / name
        trial.mkdir()
        (trial / 'lock.json').write_text(json.dumps(first))
        (trial / 'config.json').write_text(json.dumps(first))

    def check(trial):
        return {'trial': str(trial), 'trial_id': 'same' if change == 'duplicate-id' else trial.name,
                'benchmark': 'robocasa', 'task': 'CloseDrawer', 'status': 'valid_success', 'reward': 1}

    monkeypatch.setattr(check_harbor_results, 'check_trial', check)
    monkeypatch.setattr(sys, 'argv', ['check_harbor_results', str(tmp_path)])
    assert check_harbor_results.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report['valid_trials'] == report['unexpected_trials'] == 1
    assert report['missing_trial_directories'] == (0 if change == 'extra' else 1)
    assert report['task_macro_success_rate'] is report['coverage'] is None
    unexpected, = (row for row in report['trials'] if row['status'] == 'unexpected')
    assert unexpected['reward'] is None and unexpected['errors']


@pytest.mark.parametrize('legacy', [True, False])
def test_repeated_planned_identity_matches_distinct_trials(tmp_path, monkeypatch, capsys, legacy):
    import json
    import sys
    from scripts import check_harbor_results

    lock = {'task': {'name': 'close-drawer'}, 'agent': {'name': 'antigravity-cli'}}
    if not legacy:
        lock['task']['digest'] = 'sha256:aaa'
        lock['agent']['model_name'] = 'model'
    (tmp_path / 'lock.json').write_text(json.dumps({'trials': [lock, lock]}))
    for name in ('one', 'two'):
        trial = tmp_path / name
        trial.mkdir()
        (trial / 'lock.json').write_text(json.dumps(lock))
        (trial / 'config.json').write_text(json.dumps(lock))
    monkeypatch.setattr(check_harbor_results, 'check_trial', lambda trial: {
        'trial': str(trial), 'trial_id': trial.name, 'benchmark': 'robocasa', 'task': 'CloseDrawer',
        'status': 'valid_success', 'reward': 1})
    monkeypatch.setattr(sys, 'argv', ['check_harbor_results', str(tmp_path)])
    assert check_harbor_results.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report['coverage'] == report['task_macro_success_rate'] == 1
    assert report['valid_trials'] == 2 and report['unexpected_trials'] == 0


@pytest.mark.parametrize('wrong', ['task', 'agent', 'model', 'task-path'])
def test_trial_result_must_match_its_own_lock(tmp_path, wrong):
    import json
    from scripts.check_harbor_results import check_trial

    lock = {'task': {'name': 'close-drawer'}, 'agent': {'name': 'claude-code', 'model_name': 'first'}}
    config = {'agent': dict(lock['agent'])}
    result = {'task_name': 'hang-mugs' if wrong == 'task' else 'close-drawer', 'config': config}
    if wrong == 'task-path':
        lock['task'].update(type='local', path='/original/tasks/close-drawer')
        result['task_id'] = {'path': '/relocated/tasks/hang-mugs'}
    elif wrong == 'agent':
        config['agent']['name'] = 'codex'
    elif wrong == 'model':
        config['agent']['model_name'] = 'second'
    for name, data in [('lock', lock), ('config', config), ('result', result)]:
        (tmp_path / (name + '.json')).write_text(json.dumps(data))
    row = check_trial(tmp_path)
    assert row['status'] == 'invalid' and row['reward'] is None
    assert any('differs from trial lock' in error for error in row['errors'])


def test_model_slot_cannot_be_hidden_by_mislabeled_failed_probe(tmp_path, monkeypatch, capsys):
    import json
    import sys
    from scripts import check_harbor_results

    lock = {'task': {'name': 'close-drawer'}, 'agent': {'name': 'claude-code'}}
    (tmp_path / 'lock.json').write_text(json.dumps({'trials': [lock, lock]}))
    for name in ('valid', 'mislabeled'):
        trial = tmp_path / name
        trial.mkdir()
        (trial / 'lock.json').write_text(json.dumps(lock))
        (trial / 'config.json').write_text(json.dumps(lock))

    def check(trial):
        failed = trial.name == 'mislabeled'
        return {'trial': str(trial), 'diagnostic': failed, 'task': 'CloseDrawer', 'benchmark': 'robocasa',
                'status': 'invalid' if failed else 'valid_success', 'reward': None if failed else 1}

    monkeypatch.setattr(check_harbor_results, 'check_trial', check)
    monkeypatch.setattr(sys, 'argv', ['check_harbor_results', str(tmp_path)])
    assert check_harbor_results.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report['coverage'] == 0.5 and report['evaluation_trials'] == 2
    assert report['diagnostic_trials'] == 0 and report['task_macro_success_rate'] is None
