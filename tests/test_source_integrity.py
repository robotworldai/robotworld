import hashlib
import json
from pathlib import Path

import pytest

from environment.runtime.source_integrity import verify_source


def snapshot(tmp_path):
    root = tmp_path / 'benchmark'
    src = root / 'checkout'
    src.mkdir(parents=True)
    (src / 'control.py').write_text('value = 1\n')
    lock = {'commit': 'pinned-upstream', 'repository': 'https://example.org/upstream.git'}
    (root / 'project.json').write_text(json.dumps(lock))
    manifest = dict(lock, files=[{'path': 'control.py', 'sha256': hashlib.sha256(
        (src / 'control.py').read_bytes()).hexdigest()}], omissions=[], nonruntime_differences=[])
    (root / 'source-integrity.json').write_text(json.dumps(manifest))
    return root


def test_snapshot_does_not_use_parent_git(tmp_path, monkeypatch):
    root = snapshot(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('Snapshot must not use parent Git')
    monkeypatch.setattr('environment.runtime.source_integrity.subprocess.check_output', forbidden)
    result = verify_source(root)
    assert result['commit'] == 'pinned-upstream'
    assert result['upstream_modified'] is None


@pytest.mark.parametrize('change', ['modified', 'missing', 'added', 'wrong_pin', 'traversal'])
def test_snapshot_rejects_unverified_source(tmp_path, change):
    root = snapshot(tmp_path)
    if change == 'modified':
        (root / 'checkout/control.py').write_text('value = 2\n')
    elif change == 'missing':
        (root / 'checkout/control.py').unlink()
    elif change == 'added':
        (root / 'checkout/extra.py').write_text('value = 3\n')
    else:
        path = root / 'source-integrity.json'
        data = json.loads(path.read_text())
        if change == 'wrong_pin': data['commit'] = 'other'
        else: data['files'][0]['path'] = '../control.py'
        path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError):
        verify_source(root)
