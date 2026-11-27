"""Verify an independent checkout or an explicitly curated, hash-pinned source snapshot."""
import hashlib
import json
import subprocess
from pathlib import Path


def verify_source(root):
    root=Path(root);src=root/'checkout';lock=json.loads((root/'project.json').read_text())
    if (src/'.git').exists():
        head=subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()
        if head!=lock['commit']:raise RuntimeError('Wrong checkout revision: '+head)
        if subprocess.check_output(['git','-C',str(src),'status','--porcelain'],text=True).strip():
            raise RuntimeError('Expected unmodified independent upstream checkout')
        return {'commit':head,'upstream_modified':False,'source_verification':'independent_git_checkout'}
    # Never let Git walk into the RobotWorld parent repository.
    manifest=json.loads((root/'source-integrity.json').read_text())
    if manifest['commit']!=lock['commit'] or manifest['repository']!=lock['repository']:
        raise RuntimeError('Source snapshot provenance differs from project lock')
    tracked=set()
    for row in manifest['files']:
        relative=Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:raise RuntimeError('Invalid snapshot path')
        path=src/relative;tracked.add(str(relative))
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError('Source snapshot hash mismatch: '+str(relative))
    # No newly added executable Python may silently escape the manifest.
    unknown={str(p.relative_to(src)) for p in src.rglob('*.py')} - tracked
    if unknown:raise RuntimeError('Unverified executable source: '+str(sorted(unknown)[:5]))
    return {'commit':lock['commit'],'upstream_modified':None,
            'source_verification':'curated_snapshot_pinned_file_hashes',
            'source_omissions':manifest['omissions'],
            'source_nonruntime_differences':manifest['nonruntime_differences']}
