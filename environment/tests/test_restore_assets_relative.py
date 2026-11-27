import importlib.util
from pathlib import Path
import shutil
spec=importlib.util.spec_from_file_location('restore_assets',Path(__file__).resolve().parents[2]/'scripts/restore_assets.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_relative_link_survives_checkout_relocation(tmp_path):
    root=tmp_path/'old'; target=root/'Assets/test/data';target.mkdir(parents=True)
    (target/'asset').write_text('data')
    link=root/'third_party/benchmarks/test/assets'
    assert module.restore_directory_link(link,target,True)=='linked'
    assert not link.readlink().is_absolute()
    moved=tmp_path/'moved';shutil.move(root,moved)
    assert (moved/'third_party/benchmarks/test/assets/asset').read_text()=='data'


def test_migrate_only_matching_absolute_link(tmp_path):
    target=tmp_path/'Assets/data';target.mkdir(parents=True)
    link=tmp_path/'assets';link.symlink_to(target)
    assert module.restore_directory_link(link,target,True)=='relativized'
    assert str(link.readlink())=='Assets/data'
    different=tmp_path/'other';different.mkdir()
    assert module.restore_directory_link(link,different,True)=='dangling_or_different_symlink'
    assert link.resolve()==target


def test_replace_only_empty_legacy_directory(tmp_path):
    target=tmp_path/'Assets/data';target.mkdir(parents=True)
    (target/'ground.usd').write_bytes(b'original asset')
    link=tmp_path/'third_party/assets';link.mkdir(parents=True)
    assert module.restore_directory_link(link,target)=='would_link_empty_directory'
    assert not link.is_symlink()
    assert module.restore_directory_link(link,target,True)=='linked_empty_directory'
    assert (link/'ground.usd').read_bytes()==b'original asset'
    occupied=tmp_path/'occupied';occupied.mkdir();(occupied/'keep').write_text('keep')
    assert module.restore_directory_link(occupied,target,True)=='existing_legacy_directory_requires_migration'
    assert (occupied/'keep').read_text()=='keep'
