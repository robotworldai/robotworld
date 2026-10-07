#!/usr/bin/env python3
"""Verify complete pinned upstream checkout and bundled URDF/STL dependencies."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
ASSET = 'source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/assets/wheellegged_description'


def verify():
    source = ROOT / 'checkout'
    lock = json.loads((ROOT / 'project.json').read_text())
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if head != lock['commit']:
        raise RuntimeError(f'Expected {lock["commit"]}, found {head}')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Upstream checkout must remain unchanged')
    urdf = source / ASSET / 'urdf/wl_dealed.urdf'
    for mesh in ET.parse(urdf).iter('mesh'):
        dependency = (urdf.parent / mesh.attrib['filename']).resolve()
        if not dependency.is_file() or dependency.read_bytes().startswith(b'version https://git-lfs.github.com/spec'):
            raise RuntimeError(f'Missing real mesh payload (git lfs pull required): {dependency}')
    files = []
    for path in sorted((source / ASSET).rglob('*')):
        if path.is_file():
            files.append({'path': str(path.relative_to(source)), 'bytes':path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    ground = ROOT.parent / 'wheeledlab/assets/Isaac/Environments/Grid/default_environment.usd'
    return {'repository': lock['repository'], 'commit': head, 'bundled_asset_files': files,
            'robot_license': 'BSD-3-Clause; see bundled ASSET_LICENSE.md and LICENSE-BSD-3-Clause',
            'terrain': 'Native procedural plane or native mixed terrain generator; no custom replacement',
            'shared_ground_asset': {'path': str(ground), 'present': ground.is_file(),
                                    'sha256':hashlib.sha256(ground.read_bytes()).hexdigest() if ground.exists() else None}}


HDR='Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr'
HDR_SHA='1f92cbf17e46659ff37c9770d065280da226896b0f56091c4c1fca071da7fbeb'
def prepare_hdr(download=False):
    path=ROOT/'assets'/HDR
    url='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/'+HDR
    if not path.exists() and download:
        path.parent.mkdir(parents=True,exist_ok=True)
        part=path.with_suffix('.hdr.part')
        subprocess.run(['curl','-fL','--retry','2','--connect-timeout','15','--max-time','180',url,'-o',str(part)],check=True)
        part.replace(path)
    if not path.exists():raise FileNotFoundError(f'Native HDR missing: {path}; run assets with --download')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=HDR_SHA:raise RuntimeError('Official native HDR checksum mismatch')
    return {'path':'assets/'+HDR,'url':url,'sha256':digest,'bytes':path.stat().st_size}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write-manifest', action='store_true')
    p.add_argument('--download', action='store_true')
    args = p.parse_args()
    manifest = verify()
    manifest['native_hdr'] = prepare_hdr(args.download)
    if args.write_manifest:
        (ROOT / 'asset-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'commit':manifest['commit'], 'robot_asset_files':len(manifest['bundled_asset_files']),
                      'shared_ground_present':manifest['shared_ground_asset']['present']}, indent=2))

if __name__ == '__main__':
    main()
