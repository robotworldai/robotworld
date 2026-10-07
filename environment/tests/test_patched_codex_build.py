import subprocess

import pytest

from environment.runtime.patched_codex_build import PATCH_FILES, source_fingerprint


def test_patch_fingerprint_rejects_unrelated_edits_and_tracks_content(tmp_path):
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    tracked='codex-rs/core/src/context_manager/history.rs'
    path=tmp_path/tracked;path.parent.mkdir(parents=True);path.write_text('original\n')
    subprocess.run(['git','add','.'],cwd=tmp_path,check=True)
    subprocess.run(['git','-c','user.name=fixture','-c','user.email=fixture@example.invalid',
                    'commit','-qm','fixture'],cwd=tmp_path,check=True)
    for name in PATCH_FILES:
        (tmp_path/name).write_text('patched\n')
    first=source_fingerprint(tmp_path)
    path.write_text('changed patch\n')
    assert source_fingerprint(tmp_path)!=first
    (tmp_path/'unexpected.rs').write_text('unexpected\n')
    with pytest.raises(ValueError,match='Unexpected changes'):
        source_fingerprint(tmp_path)
