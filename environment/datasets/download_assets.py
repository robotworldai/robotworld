"""Fetch exactly the manifest-listed assets from a Hugging Face dataset repo."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile


def valid_file(path, entry):
    if not path.is_file() or path.stat().st_size != entry['bytes']:
        return False
    with path.open('rb') as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == entry['sha256']


def relative_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value:
        raise ValueError(f'Unsafe manifest path: {value}')
    return path


def fetch_assets(manifest, destination, cache, repo_id=None, revision=None,
                 verify_only=False, downloader=None):
    destination, cache = Path(destination).resolve(), Path(cache).resolve()
    prefix = relative_path(manifest['prefix'])
    entries = manifest['files']
    if len(entries) != manifest['file_count'] or sum(x['bytes'] for x in entries) != manifest['total_bytes']:
        raise ValueError('Manifest count/size mismatch')
    if len({e['path'] for e in entries}) != len(entries):
        raise ValueError('Duplicate manifest paths')
    pending = []
    for entry in entries:
        rel = relative_path(entry['path'])
        target = destination / str(rel)
        if not target.resolve().is_relative_to(destination):
            raise ValueError('Asset path escapes destination through symlink')
        if not valid_file(target, entry):
            pending.append((entry, rel, target))
    if verify_only:
        if pending:
            raise RuntimeError(f'{len(pending)} missing or invalid assets; first: {pending[0][0]["path"]}')
        return {'verified': len(entries), 'downloaded': 0}
    repo_id = repo_id or manifest.get('repo_id')
    revision = revision or manifest.get('revision') or 'main'
    if pending and not repo_id:
        raise ValueError('HF repository is not configured yet; pass --repo-id OWNER/REPO')
    if pending and downloader is None:
        from huggingface_hub import hf_hub_download
        downloader = hf_hub_download
    for index, (entry, rel, target) in enumerate(pending, 1):
        print(f'[{index}/{len(pending)}] {rel}', flush=True)
        downloaded = Path(downloader(repo_id=repo_id, repo_type='dataset',
                                    revision=revision, filename=str(prefix / rel),
                                    local_dir=str(cache)))
        if not valid_file(downloaded, entry):
            # A wrong revision/corrupt cache must never replace the installed file.
            raise RuntimeError(f'Asset size/SHA256 mismatch: {rel}')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.asset-', delete=False) as handle:
                temporary = Path(handle.name)
                with downloaded.open('rb') as source:
                    shutil.copyfileobj(source, handle)
            os.replace(temporary, target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return {'verified': len(entries), 'downloaded': len(pending)}


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).parent / 'manifests/robodojo-conveyor.json')
    parser.add_argument('--repo-id', help='HF dataset repo, e.g. OWNER/world-assets')
    parser.add_argument('--revision', help='Prefer the full HF commit hash after publishing')
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    destination = args.destination or root / str(relative_path(manifest['destination']))
    result = fetch_assets(manifest, destination, root / 'var/cache/hf-assets',
                          args.repo_id, args.revision, args.verify_only)
    print(json.dumps({**result, 'destination': str(destination)}, indent=2))


if __name__ == '__main__':
    main()
