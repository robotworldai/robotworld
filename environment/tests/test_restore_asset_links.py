import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def restore(tmp_path,monkeypatch):
    script=Path(__file__).resolve().parents[2]/'scripts/restore_assets.py'
    spec=importlib.util.spec_from_file_location('restore_assets_under_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    layout=tmp_path/'environment/datasets';layout.mkdir(parents=True)
    (layout/'asset-layout.json').write_text(json.dumps({'external':[
        {'source':'var/datasets/demo','target':'Assets/demo/data'}],
        'embedded_manifest':'embedded.json'}))
    (tmp_path/'embedded.json').write_text('{"files":[]}')
    (tmp_path/'Assets/demo/data').mkdir(parents=True)
    actual=tmp_path/'external-runtime'/'nested';actual.mkdir(parents=True)
    (tmp_path/'var').symlink_to(actual,target_is_directory=True)
    monkeypatch.setattr(sys,'argv',['restore_assets','--apply','--bench','demo'])
    return module,tmp_path


def test_link_resolves_through_external_var_symlink(restore):
    module,root=restore
    module.main()
    link=root/'var/datasets/demo'
    assert link.resolve()==root/'Assets/demo/data'
    assert link.exists()
    module.main()


def test_dangling_link_is_reported_without_overwrite(restore):
    module,root=restore
    link=root/'var/datasets/demo';link.parent.mkdir()
    link.symlink_to('../../Assets/demo/data')
    with pytest.raises(SystemExit):module.main()
    assert link.readlink()==Path('../../Assets/demo/data')
    report=json.loads((root/'var/asset-restore.json').read_text())
    assert report['files'][0]['status']=='dangling_or_different_symlink'


def test_restore_preserves_checkout_mode_without_mutating_asset(restore):
    module,root=restore
    asset=root/'original-asset';asset.write_bytes(b'actual');asset.chmod(0o644)
    source=root/'tracked';source.write_bytes(b'pointer');source.chmod(0o755)
    module.link_atomically(asset,source)
    assert source.read_bytes()==b'actual'
    assert source.stat().st_mode & 0o777 == 0o755
    assert asset.stat().st_mode & 0o777 == 0o644
    assert source.stat().st_ino != asset.stat().st_ino
