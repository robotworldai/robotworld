"""Validate trusted sidecar evidence and report confirmed task completion.

This verifier does not infer physical success from pixels or agent statements.
Invalid evidence produces details.json and a nonzero exit, never reward=0.
A normal early return without confirmed completion receives zero; an unavailable
native score remains unavailable. Probe agents are excluded by the host audit.
"""
import argparse
import json
import math
import subprocess
from pathlib import Path


class InvalidTrial(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidTrial(message)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_feedback_callbacks(commands, receipt, callbacks, actions, variable):
    # program_step starts at zero for each coding_control command.
    groups = []
    message = 'Feedback callback count contradicts receipt or request'
    for step, done in callbacks:
        require(type(step) is int, message)
        if step == 0:
            groups.append((0, False))
        require(groups and groups[-1] == (step, False), message)
        groups[-1] = (step + 1, done)

    counts = receipt['executed_steps_per_command']
    programs = [(i, cmd['arguments']['max_steps']) for i, cmd in enumerate(commands)
                if cmd.get('tool') == 'coding_control']
    if not programs:
        require(not callbacks, message)
        return
    if 'feedback' not in receipt:
        # Older receipts omit program errors and per-command planner calls.
        minimum = sum(int(counts[i] > 0) if variable else counts[i] for i, _ in programs)
        require(len(groups) <= len(programs) and minimum <= len(callbacks) <= sum(n for _, n in programs), message)
        return

    errors = {item['command_index'] for item in receipt['feedback'] if 'program_error' in item['details']}
    last_action = max((i for i, n in enumerate(actions) if n), default=-1)
    # A worker can fail before producing any journal event. Adjacent programs
    # can then share the same candidate first event until one consumes it.
    positions = {0}
    for i, limit in programs:
        if receipt['terminal'] and i > last_action:
            break
        count = actions[i]
        done = i not in errors and count < limit and not (receipt['terminal'] and i == last_action)
        lengths = {count + int(done)}
        if i in errors and count < limit:
            # An invalid returned command is journaled; a worker failure isn't.
            lengths.add(count + 1)
        require(count <= limit, message)
        matches = {pos + 1 for pos in positions if pos < len(groups)
                   and groups[pos][1] == done and groups[pos][0] in lengths}
        if 0 in lengths:
            matches.update(positions)
        positions = matches
    require(len(groups) in positions, message)


def verify_video(path, expected_frames, frequency):
    probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames,avg_frame_rate,width,height', '-of', 'json', str(path)],
        capture_output=True, text=True, timeout=30, check=True)
    streams = json.loads(probe.stdout)['streams']
    require(len(streams) == 1, 'Missing video stream')
    stream = streams[0]
    frames = int(stream['nb_read_frames'])
    require(frames > 0, 'Empty video')
    if expected_frames is not None:
        require(frames == expected_frames, 'Video frame count must equal initial frame plus every control step')
    num, den = map(float, stream['avg_frame_rate'].split('/'))
    require(den > 0, 'Invalid video frame rate')
    if frequency is not None:
        require(math.isclose(num / den, frequency, rel_tol=0.001), 'Video frequency differs from native control frequency')
    require(stream['width'] > 0 and stream['height'] > 0, 'Invalid video dimensions')
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-map', '0:v:0', '-f', 'null', '-'],
                   capture_output=True, timeout=30, check=True)
    return {'frames': frames, 'fps': num / den, 'decoded': True}


def audit_artifacts(runs, expected=None):
    runs = Path(runs)
    if expected is None:
        config_path = Path(__file__).with_name('task.json')
        expected = json.loads(config_path.read_text()) if config_path.exists() else {'benchmark':'robocasa','task':'CloseDrawer','action_dimension':12}
    require(not (runs / 'error.txt').exists(), 'Simulator recorded an exception')
    require((runs / 'finalized').read_text().strip() == 'complete', 'Simulator was not finalized')
    result = json.loads((runs / 'episode.json').read_text())
    config_file = runs / 'service-config.json'
    if not config_file.exists():
        config_file = runs / 'evaluation-config.json'
    config = json.loads(config_file.read_text())
    require(result['protocol'] == 'robotworld.episode.v1', 'Protocol mismatch')
    for key in ('benchmark', 'task'):
        require(config.get(key) == expected[key], 'Wrong task evidence: ' + key)
    steps, horizon = result['control_step'], result['horizon']
    require(type(steps) is int and type(horizon) is int and 0 <= steps <= horizon, 'Invalid step budget')
    require(horizon == config['native_horizon'] and horizon > 0, 'Native budget mismatch')
    if expected.get('horizon') is not None:
        require(horizon == expected['horizon'], 'Task budget differs from verifier specification')
    require(result['terminal'] is True and type(result['success']) is bool, 'Missing terminal result')
    require(result['observation_id'] == f"{result['episode_id']}:{steps}", 'Observation ID mismatch')
    execution_path = runs / 'execution-config.json'
    execution_config = json.loads(execution_path.read_text()) if execution_path.exists() else {}
    cap = execution_config.get('control_step_limit')
    if cap is not None:
        require(type(cap) is int and cap > 0, 'Invalid simulator-owned control-step cap')
        require(execution_config.get('episode_id') == result['episode_id'] and execution_config.get('native_horizon') == horizon, 'Wrong execution configuration')
        require(result.get('control_step_limit') == cap, 'Control-step cap mismatch')
        require(result.get('effective_control_step_limit') == min(cap,horizon), 'Effective cap mismatch')
        require(result.get('control_step_limit_overshoot') == max(0,steps-cap), 'Cap overshoot mismatch')
    else:
        require(result.get('control_step_limit') is None and result.get('stop_reason') != 'control_step_limit', 'Unconfigured cap claimed')
    reason = result['termination']
    require(reason in {'success', 'native_failure', 'native_truncation', 'budget_exhausted', 'agent_stopped'}, 'Infrastructure or unknown termination')
    trace_path = runs / 'native-steps.jsonl'
    # Some adapters open their native trace on the first physical action.
    native = read_jsonl(trace_path) if steps or trace_path.exists() else []
    variable = expected['benchmark'] == 'robodojo'
    planner_index = 0
    if not variable:
        require(len(native) == steps, 'Native trace length mismatch')
    previous = 0
    by_step = {}
    for number, row in enumerate(native, 1):
        if variable:
            require(row['control_step_before'] == previous and row['control_step'] >= previous, 'Non-contiguous planned-motion spans')
            require(row['executed_steps'] == row['control_step'] - previous, 'Motion span count mismatch')
        else:
            require(type(row['control_step']) is int and row['control_step'] == number, 'Non-contiguous native steps')
            action = row['action']
            require(isinstance(action,list) and action and all(type(x) in (int,float) and math.isfinite(x) for x in action), 'Invalid native action')
            if expected.get('action_dimension') is not None:
                require(len(action) == expected['action_dimension'], 'Native action dimension mismatch')
        require(all(type(row[k]) is bool for k in ('success','terminated','truncated')), 'Invalid native flags')
        if number < len(native):
            require(not any(row[k] for k in ('success','terminated','truncated')), 'Execution continued after native terminal')
        previous = row['control_step']
        by_step[previous] = row
    require(previous == steps, 'Final native step mismatch')
    last = native[-1] if native else dict(success=False,terminated=False,truncated=False)
    require(result['success'] == last['success'], 'Native success mismatch')
    expected_reason = ('success' if last['success'] else 'native_failure' if last['terminated'] else
                       'native_truncation' if last['truncated'] else 'budget_exhausted' if steps == horizon else 'agent_stopped')
    require(reason == expected_reason, 'Terminal reason contradicts native state/budget')
    if cap is not None:
        expected_stop_reason = 'control_step_limit' if reason == 'agent_stopped' and steps >= cap else None
        require(result.get('stop_reason') == expected_stop_reason, 'Cap stop reason contradicts terminal state')
    pending = None
    callbacks = []
    completed_ids = set()
    total = 0
    previous_terminal = False
    for event in read_jsonl(runs / 'events.jsonl'):
        data = event['data']
        if event['event'] == 'feedback_callback':
            # Callbacks have no episode ID; their enclosing request supplies it.
            require(pending is not None and any(c.get('tool') == 'coding_control' for c in pending['commands']),
                    'Feedback callback outside coding_control request')
            callbacks.append((data.get('program_step'), data.get('reply', {}).get('action') == {'done': True}))
            continue
        require(data['episode_id'] == result['episode_id'], 'Journal episode mismatch')
        if event['event'] == 'execution_requested':
            require(not previous_terminal, 'Execution continued after terminal receipt')
            require(pending is None and data['request_id'] not in completed_ids, 'Repeated or overlapping execution')
            require(data['observation_id'] == f"{result['episode_id']}:{total}", 'Stale journal observation')
            if execution_config.get('code_control') is False:
                require(not any(cmd.get('tool') == 'coding_control' for cmd in data['commands']),
                        'Feedback control disabled for this episode')
            pending = data
            callbacks = []
        elif event['event'] == 'execution_completed':
            require(pending is not None and pending['request_id'] == data['request_id'], 'Unmatched receipt')
            require(data['error'] is None and data['execution_uncertain'] is False, 'Uncertain execution')
            counts = data['executed_steps_per_command']
            require(len(counts) == len(pending['commands']), 'Receipt command count mismatch')
            require(all(type(n) is int and n >= 0 for n in counts), 'Invalid executed count')
            actions = counts
            if variable and 'feedback' in data:
                # Every planner invocation records one motion even at zero ticks.
                actions = [0] * len(counts)
                for item in data['feedback']:
                    if 'motion_accepted' in item['details']:
                        i = item['command_index']
                        require(type(i) is int and 0 <= i < len(actions), 'Unknown planner command')
                        actions[i] += 1
                if planner_index is not None:
                    for calls, ticks in zip(actions, counts):
                        motions = native[planner_index:planner_index + calls]
                        require(len(motions) == calls and sum(m['executed_steps'] for m in motions) == ticks,
                                'Planner motion count mismatch')
                        planner_index += calls
            elif variable:
                # Older receipts cannot resolve consecutive motions at the same step.
                planner_index = None
            if not variable:
                for count, command in zip(counts, pending['commands']):
                    key = 'max_steps' if command.get('tool') == 'coding_control' else 'steps'
                    require(count <= command['arguments'][key], 'Invalid executed count')
            if callbacks or execution_config.get('code_control') is True:
                audit_feedback_callbacks(pending['commands'], data, callbacks, actions, variable)
            require(sum(counts) == data['executed_steps'], 'Receipt count mismatch')
            total += sum(counts)
            require(data['control_step'] == total and total <= steps, 'Receipt step mismatch')
            require(data['horizon'] == horizon, 'Receipt budget mismatch')
            if variable and planner_index is not None:
                row = native[planner_index-1] if planner_index else {}
            else:
                row = by_step.get(total, {})
            row = row or dict(success=False, terminated=False, truncated=False)
            require(data['success'] == row['success'], 'Receipt success mismatch')
            receipt_reason = ('success' if row['success'] else 'native_failure' if row['terminated'] else
                              'native_truncation' if row['truncated'] else 'budget_exhausted' if total == horizon else None)
            if cap is not None:
                require(data.get('control_step_limit') == cap, 'Receipt cap mismatch')
                require(data.get('control_step_limit_overshoot') == max(0,total-cap), 'Receipt cap overshoot mismatch')
                require(variable or total <= cap, 'Non-macro action exceeded cap')
                if receipt_reason is None and total >= cap:
                    receipt_reason = 'agent_stopped'
                    require(data.get('stop_reason') == 'control_step_limit', 'Missing cap stop reason')
                else:
                    require(data.get('stop_reason') is None, 'Unexpected cap stop reason')
            require(data['termination'] == receipt_reason and data['terminal'] == (receipt_reason is not None), 'Receipt terminal state mismatch')
            require(data['observation_id'] == f"{result['episode_id']}:{total}", 'Receipt observation mismatch')
            previous_terminal = data['terminal']
            completed_ids.add(data['request_id'])
            pending = None
        else:
            raise InvalidTrial('Unknown journal event')
    require(pending is None and total == steps, 'Incomplete execution journal')
    if variable and planner_index is not None:
        require(planner_index == len(native), 'Unaccounted planner motion')
    # Episode state is cross-checked with the original scorer, never with agent text.
    native_result = None
    for filename in ('result.json','native-result.json','native-results.json'):
        path = runs / filename
        if path.exists():
            native_result = json.loads(path.read_text())
            if isinstance(native_result,list):
                require(len(native_result) <= 1, 'Expected single environment result')
                native_result = native_result[0] if native_result else None
            break
    diagnostic = reason == 'agent_stopped'
    if expected['benchmark'] != 'robocasa':
        if native_result is None:
            require(diagnostic, 'Missing authoritative scorer result')
        elif native_result.get('success') is None:
            require(diagnostic, 'Native success is unavailable; cannot invent binary score')
        else:
            require(type(native_result['success']) is bool and native_result['success'] == result['success'], 'Authoritative scorer success mismatch')
        configured_profile = expected.get('scoring_profile')
        if configured_profile == 'world-state-v1' and native_result is not None:
            require(native_result.get('scoring_profile') == configured_profile, 'Wrong scoring profile')
            require(not native_result.get('world_evaluation',{}).get('reader_errors'), 'Scorer could not read required state')
    strict = expected['benchmark'] == 'robocasa'
    videos = []
    paths = [p for p in (runs/'video.mp4',runs/'episode.mp4') if p.is_file()]
    for folder in ('video','videos'):
        if (runs/folder).exists():
            paths.extend(sorted((runs/folder).rglob('*.mp4')))
    require(bool(paths), 'Missing recorded video')
    for path in paths:
        if strict:
            checked = verify_video(path,steps+1,config['control_frequency_hz'])
        else:
            checked = verify_video(path,None,config.get('control_frequency_hz'))
            # Native encoders may include or omit the initial frame. No sampling is accepted by default.
            if not variable and expected.get('video_frame_policy','every_step') == 'every_step':
                require(checked['frames'] >= max(1,steps), 'Video omits control steps')
        videos.append({'path':str(path.relative_to(runs)),**checked})
    native_success = (native_result.get('success') if native_result is not None else
                      last['success'] if expected['benchmark'] == 'robocasa' else None)
    return {'status':'valid_success' if result['success'] else 'valid_failure',
            'reward':int(result['success']), 'episode':result,
            'native_success':native_success,
            'score_basis':('goal_confirmed_by_evaluator' if result['success'] else
                           'goal_not_confirmed_before_control_step_limit' if result.get('stop_reason') == 'control_step_limit' else
                           'goal_not_confirmed_before_agent_return' if diagnostic else
                           'goal_not_completed_before_environment_end'),
            'note':'Completion reward is not a fabricated native score. Host auditing excludes diagnostic agents and invalid execution trials.',
            'video':videos[0] if len(videos)==1 else videos,
            'success_source':'trusted simulator native/World scorer; not agent self-report',
            'checks':['finalized','task_identity','native_steps','execution_receipts','terminal_budget','native_scorer','video_decode']}



def validate(runs, expected=None):
    """Score only the selected environment outcome; artifact QA is separate."""
    runs = Path(runs)
    if expected is None:
        expected = json.loads(Path(__file__).with_name('task.json').read_text())
    result = json.loads((runs / 'episode.json').read_text())
    config_path = runs / 'service-config.json'
    if not config_path.exists(): config_path = runs / 'evaluation-config.json'
    config = json.loads(config_path.read_text())
    for key in ('benchmark', 'task'):
        require(config.get(key) == expected[key], 'Wrong task result: ' + key)
    require(result.get('terminal') is True, 'Environment has no final outcome')
    reason = result.get('termination')
    require(reason in {'success','native_failure','native_truncation','budget_exhausted','agent_stopped'},
            'Environment error or unknown outcome')
    require(type(result.get('success')) is bool, 'Missing environment success flag')
    require((reason == 'success') == result['success'], 'Inconsistent environment outcome')
    # episode.json is the authoritative simulator-service outcome. Auxiliary
    # native result files and physical evidence are diagnostics only: do not
    # re-evaluate or override the environment here.
    return {'status':'valid_success' if result['success'] else 'valid_failure',
            'reward':int(result['success']), 'episode':result,
            'environment_success':result['success'], 'outcome_file':'episode.json', 'video':None,
            'score_basis':'configured_environment_outcome',
            'success_source':'simulator native/World checker, not agent self-report',
            'checks':['task_identity','environment_final_outcome'],
            'artifact_audit_required_for_reward':False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, default=Path('/runs'))
    parser.add_argument('--spec', type=Path, help='Trusted task verifier specification')
    parser.add_argument('--output', type=Path, default=Path('/logs/verifier'))
    parser.add_argument('--audit-only', action='store_true', help='Optional artifact QA; never writes or removes reward')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        try:
            report = audit_artifacts(args.runs, json.loads(args.spec.read_text()) if args.spec else None)
        except Exception as exc:
            report = {'status':'audit_failed', 'error_type':type(exc).__name__, 'error':str(exc)}
        (args.output / 'artifact-audit.json').write_text(json.dumps(report, indent=2))
        return
    (args.output / 'reward.txt').unlink(missing_ok=True)
    try:
        details = validate(args.runs, json.loads(args.spec.read_text()) if args.spec else None)
    except Exception as exc:
        (args.output / 'details.json').write_text(json.dumps({'status': 'invalid', 'error_type': type(exc).__name__, 'error': str(exc)}, indent=2))
        raise
    (args.output / 'details.json').write_text(json.dumps(details, indent=2))
    (args.output / 'reward.txt').write_text(f"{details['reward']}\n")


if __name__ == '__main__':
    main()
