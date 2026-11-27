"""Restore the eight pinned runtime assets from their NVIDIA source URLs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener


def valid(path, row):
    if not path.is_file() or path.stat().st_size != row['bytes']:
        return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256']


def restore(destination, row):
    if destination.exists():
        if not valid(destination, row):
            raise ValueError(f'Existing asset differs: {destination}')
        return
    url = urlsplit(row['source_url'])
    if url.scheme != 'https' or url.hostname not in {
        'omniverse-content-production.s3.us-west-2.amazonaws.com',
        'omniverse-content-production.s3-us-west-2.amazonaws.com',
    }:
        raise ValueError('Unexpected upstream asset URL')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        path = Path(temporary.name)
    try:
        with build_opener(ProxyHandler({})).open(url.geturl(), timeout=90) as source, path.open('wb') as sink:
            shutil.copyfileobj(source, sink)
        if not valid(path, row):
            raise ValueError(f'Upstream checksum mismatch: {destination}')
        path.chmod(0o644)
        path.replace(destination)
    finally:
        path.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path, help='RobotWorld checkout containing Assets/')
    parser.add_argument('--bench', choices=['robolab', 'omnidrones'])
    args = parser.parse_args()
    for row in json.loads(args.manifest.read_text()):
        relative = Path(row['path'])
        if len(relative.parts) < 3 or relative.parts[0] != 'Assets' or '..' in relative.parts:
            raise ValueError('Invalid asset path')
        if not args.bench or relative.parts[1] == args.bench:
            restore(args.root / relative, row)
            print(relative)
