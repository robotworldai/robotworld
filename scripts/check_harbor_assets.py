"""Reject asset roots whose top-level links cannot survive the Docker mount."""
import argparse
from collections import deque
from pathlib import Path


def _portable_link(root, child):
    if not child.exists():
        return False
    # Follow each component as it will resolve after the root is relocated.
    # A link that leaves the mount and re-enters it only works on the host.
    current = root
    pending = deque([child.name])
    while pending:
        part = pending.popleft()
        if part == '..':
            if current == root:
                return False
            current = current.parent
            continue
        current = current / part
        if current.is_symlink():
            target = current.readlink()
            if target.is_absolute():
                return False
            pending.extendleft(reversed(target.parts))
            current = current.parent
    return True


def validate_asset_root(path):
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError('WORLD_ASSETS must be a directory')
    invalid = []
    for child in root.iterdir():
        if child.is_symlink() and not _portable_link(root, child):
            invalid.append(child.name)
    if invalid:
        raise ValueError('WORLD_ASSETS contains non-portable top-level symlinks: '
                         + ', '.join(sorted(invalid))
                         + '. Set WORLD_ASSETS to the actual asset directory, not a directory of host symlinks.')
    return root


def prepare_asset_mount(path, checkout):
    root = validate_asset_root(path)
    # Docker cannot create a nested mountpoint inside the read-only checkout.
    (Path(checkout) / 'Assets').mkdir(exist_ok=True)
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path')
    parser.add_argument('--prepare-mount', metavar='CHECKOUT', type=Path,
                        help='Create the Assets mountpoint inside this checkout after validation')
    args = parser.parse_args()
    try:
        if args.prepare_mount:
            print(prepare_asset_mount(args.path, args.prepare_mount))
        else:
            print(validate_asset_root(args.path))
    except (OSError, ValueError) as exc:
        parser.exit(2, str(exc) + '\n')
