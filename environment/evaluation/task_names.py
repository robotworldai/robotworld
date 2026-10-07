"""Public descriptive names; legacy task IDs remain private routing aliases."""
from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=1)
def entries():
    data = json.loads(Path(__file__).with_name('native17.json').read_text())
    return tuple(data['tasks'] + data.get('variants', []))


def identity(benchmark, task):
    for row in entries():
        if row['project'] == benchmark and task in (row['id'], row['name']):
            return row
    return None


def canonical(benchmark, task):
    row = identity(benchmark, task)
    return row['name'] if row else task


def routing_key(benchmark, task):
    row = identity(benchmark, task)
    return row['id'] if row and not row['existing_integration'] else canonical(benchmark, task)


def title(benchmark, task):
    row = identity(benchmark, task)
    return row['title'] if row else task


def named_record(record):
    """Normalize metadata, never modify native raw results or rewrite historical commands."""
    record = dict(record)
    bench = record.get('benchmark')
    task = record.get('task')
    row = identity(bench, task)
    if row:
        record.update(task=row['name'], task_title=row['title'], legacy_task_id=row['id'])
    return record
