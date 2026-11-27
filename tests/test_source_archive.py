import hashlib,json
import pytest
from environment.runtime.source_archive import verify_archive

def test_archive_without_git_is_verified(tmp_path):
 root=tmp_path/'src';root.mkdir();(root/'a.py').write_text('x=1\n')
 manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'commit':'pinned','files':{'a.py':hashlib.sha256((root/'a.py').read_bytes()).hexdigest()}}))
 assert verify_archive(root,manifest,'pinned')=='pinned'
 (root/'a.py').write_text('x=2\n')
 with pytest.raises(RuntimeError,match='changed'):verify_archive(root,manifest,'pinned')

def test_archive_rejects_untracked_python(tmp_path):
 root=tmp_path/'src';root.mkdir();(root/'a.py').write_text('');(root/'extra.py').write_text('')
 manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'commit':'pinned','files':{'a.py':hashlib.sha256(b'').hexdigest()}}))
 with pytest.raises(RuntimeError,match='Unpinned'):verify_archive(root,manifest,'pinned')
