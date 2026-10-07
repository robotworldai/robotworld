"""Resume orchestration only; never restore simulator or agent state."""
from contextlib import contextmanager
from .task_names import canonical, named_record
import fcntl
import json
from pathlib import Path


@contextmanager
def batch_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'Batch is already being processed: {path}') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def refresh(record):
    path = Path(record['path'])/'run.json'
    return named_record(json.loads(path.read_text()) if path.exists() else dict(record))


def resume_plans(plans, prior):
    """Keep completed logical rollouts; retain incomplete attempts under separate paths."""
    old = {}
    for r in prior['runs']:
        r = refresh(r)
        old[r['task'], r['rollout']] = r
    if set(old) != {(r['task'], r['rollout']) for _, r, _ in plans}:
        raise ValueError('Resume must use the same tasks and rollout count as the original batch')
    result = []
    for task, record, seed in plans:
        previous = old[record['task'], record['rollout']]
        if previous.get('scoring_profile','native') != record.get('scoring_profile','native'):
            raise ValueError(f"{record['task']}: resume scoring profile differs; use a new batch")
        fields = ('model','step_limit','runtime','seed','code_control_requested','scene_config')
        for field in fields:
            if previous.get(field) != record.get(field):
                raise ValueError(f"{record['task']}: resume {field} differs; use a new batch")
        if previous['status'] == 'completed':
            # A finished native failure is also a completed measurement, not a retry candidate.
            result.append((task, previous, seed))
            continue
        history = list(previous.get('previous_attempts', []))
        old_path = Path(previous['path'])
        record['path'] = str(old_path)  # A planned retry may not have started before interruption.
        if old_path.exists():
            history.append({k: previous.get(k) for k in (
                'path','status','started_at','finished_at','error','code_control_effective')})
            base = Path(record['path'])
            attempt = len(history)
            candidate = base.with_name(base.name+f'-retry-{attempt:02d}')
            while candidate.exists():
                attempt += 1
                candidate = base.with_name(base.name+f'-retry-{attempt:02d}')
            record['path'] = str(candidate)
        record['previous_attempts'] = history
        result.append((task, record, seed))
    return result


def campaign_record(row, output_root, batch):
    """Find the latest attempt through the benchmark summary, then trust its run.json."""
    manifest = Path(output_root)/row['benchmark']/'summaries'/f'{batch}.json'
    record = {'path': row['run'], 'status': 'not_run'}
    if manifest.exists():
        records = json.loads(manifest.read_text())['runs']
        record = next(r for r in records if canonical(row['benchmark'], r['task']) == canonical(row['benchmark'], row['task']) and r['rollout'] == 1)
    record = refresh(record)
    row['run'] = record['path']
    return record
