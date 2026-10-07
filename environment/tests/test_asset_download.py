import hashlib
from pathlib import Path
import pytest
from environment.datasets.download_assets import fetch_assets


def manifest():
    return {'prefix': 'robodojo/scene/Assets', 'file_count': 1, 'total_bytes': 3,
            'files': [{'path': 'Robot/x.usd', 'bytes': 3,
                       'sha256': hashlib.sha256(b'abc').hexdigest()}]}


def test_download_hash_and_skip(tmp_path):
    calls = []
    def download(**kw):
        calls.append(kw)
        p = tmp_path / 'downloaded'
        p.write_bytes(b'abc')
        return str(p)
    dest = tmp_path / 'assets'
    result = fetch_assets(manifest(), dest, tmp_path / 'cache', 'owner/repo', 'pinned', downloader=download)
    assert result == {'verified': 1, 'downloaded': 1}
    assert calls[0]['filename'] == 'robodojo/scene/Assets/Robot/x.usd'
    assert calls[0]['repo_type'] == 'dataset' and calls[0]['revision'] == 'pinned'
    assert fetch_assets(manifest(), dest, tmp_path / 'cache', verify_only=True)['verified'] == 1
    fetch_assets(manifest(), dest, tmp_path / 'cache', downloader=download)
    assert len(calls) == 1


def test_bad_download_preserves_existing_file(tmp_path):
    target = tmp_path / 'assets/Robot/x.usd'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'old')
    bad = tmp_path / 'bad'
    bad.write_bytes(b'bad')
    with pytest.raises(RuntimeError, match='SHA256'):
        fetch_assets(manifest(), tmp_path / 'assets', tmp_path / 'cache', 'owner/repo', downloader=lambda **kw: bad)
    assert target.read_bytes() == b'old'


def test_rejects_manifest_escape_and_missing_repo(tmp_path):
    m = manifest()
    with pytest.raises(ValueError, match='--repo-id'):
        fetch_assets(m, tmp_path / 'assets', tmp_path / 'cache')
    m['files'][0]['path'] = '../outside'
    with pytest.raises(ValueError, match='Unsafe'):
        fetch_assets(m, tmp_path / 'assets', tmp_path / 'cache')
    m = manifest()
    (tmp_path / 'assets').mkdir()
    (tmp_path / 'elsewhere').mkdir()
    (tmp_path / 'assets/Robot').symlink_to(tmp_path / 'elsewhere', target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        fetch_assets(m, tmp_path / 'assets', tmp_path / 'cache')
