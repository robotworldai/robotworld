from pathlib import Path
import pytest
from scripts.check_harbor_assets import prepare_asset_mount, validate_asset_root


def test_root_link_resolves_before_mount(tmp_path):
    actual = tmp_path / 'actual'; actual.mkdir()
    alias = tmp_path / 'alias'; alias.symlink_to(actual)
    assert validate_asset_root(alias) == actual


def test_absolute_child_link_rejected_even_when_valid_on_host(tmp_path):
    root = tmp_path / 'assets'; root.mkdir()
    target = tmp_path / 'actual'; target.mkdir()
    (root / 'bench').symlink_to(target)
    with pytest.raises(ValueError, match='non-portable'):
        validate_asset_root(root)


def test_relative_internal_link_allowed(tmp_path):
    (tmp_path / 'shared').mkdir()
    (tmp_path / 'bench').symlink_to('shared')
    assert validate_asset_root(tmp_path) == tmp_path


def test_broken_or_escaping_link_rejected(tmp_path):
    root = tmp_path / 'assets'; root.mkdir()
    (root / 'bench').symlink_to('../missing')
    with pytest.raises(ValueError, match='non-portable'):
        validate_asset_root(root)


def test_link_that_leaves_and_reenters_root_rejected(tmp_path):
    root = tmp_path / 'storage'
    (root / 'data').mkdir(parents=True)
    (root / 'bench').symlink_to('../storage/data')
    with pytest.raises(ValueError, match='bench'):
        validate_asset_root(root)


def test_portability_checked_through_nested_link(tmp_path):
    nested = tmp_path / 'nested'
    nested.mkdir()
    (tmp_path / 'data').mkdir()
    (tmp_path / 'bench').symlink_to('nested/target')
    target = nested / 'target'
    target.symlink_to('../data')
    assert validate_asset_root(tmp_path) == tmp_path
    target.unlink()
    target.symlink_to(tmp_path / 'data')
    with pytest.raises(ValueError, match='bench'):
        validate_asset_root(tmp_path)


def test_prepare_creates_mountpoint_only_after_validation(tmp_path):
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    mountpoint = checkout / 'Assets'
    with pytest.raises(FileNotFoundError):
        prepare_asset_mount(mountpoint, checkout)
    assert not mountpoint.exists()

    external = tmp_path / 'storage'
    external.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(external)
    assert validate_asset_root(alias) == external
    assert not mountpoint.exists()
    assert prepare_asset_mount(alias, checkout) == external
    assert mountpoint.is_dir() and not list(mountpoint.iterdir())
    assert prepare_asset_mount(alias, checkout) == external


def test_prepare_preserves_existing_asset_directory_symlink(tmp_path):
    external = tmp_path / 'storage'
    external.mkdir()
    mountpoint = tmp_path / 'Assets'
    mountpoint.symlink_to(external)
    assert prepare_asset_mount(mountpoint, tmp_path) == external
    assert mountpoint.is_symlink()
