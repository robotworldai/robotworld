"""Verify upstream resources; missing author payloads cannot be substituted."""
import hashlib
import json
from pathlib import Path


def verify():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'asset-manifest.json').read_text())
    for row in manifest['files']:
        path=root/'checkout'/row['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError('Missing/changed official resource: '+str(path))
    missing=[p for p in manifest['missing_author_payloads'] if not Path(p).is_file()]
    if missing:
        raise RuntimeError('Original asset closure incomplete; obtain author payloads, not replacement models:\n'+'\n'.join(missing))
    return manifest


if __name__=='__main__':
    verify()
    print('Pinned local files verified; runtime USD resolution still needs probe.')
