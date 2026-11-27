"""Verify a distributed source tree against a trusted pinned-file manifest."""
import hashlib
import json
from pathlib import Path


def verify_archive(root, manifest, expected_commit):
    root=Path(root).resolve(); data=json.loads(Path(manifest).read_text())
    if data.get('commit')!=expected_commit or not data.get('files'):
        raise RuntimeError('Source manifest does not match the required commit')
    for relative,digest in data['files'].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError('Missing or escaping pinned source file: '+relative)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Pinned source file changed: '+relative)
    # Extra Python modules can change import resolution even if tracked files match.
    unexpected={str(p.relative_to(root)) for p in root.rglob('*.py')}-data['files'].keys()
    if unexpected:raise RuntimeError('Unpinned Python source: '+', '.join(sorted(unexpected)[:5]))
    return data['commit']
