"""Check actual robot/table USD reference closure and cache exact official environment resources."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urljoin

ROOT=Path(__file__).resolve().parent
BASE='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/'
NATIVE_SEEDS=['legged_lab/assets/booster/T1_TT/T1_TT.usd',
              'legged_lab/assets/table_tennis/table/pp_table_ver2.usd']
OFFICIAL_SEEDS=['Isaac/Environments/Grid/default_environment.usd',
    'Isaac/IsaacLab/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl',
    'Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr']
BUILTIN_MDL={'OmniPBR.mdl','OmniGlass.mdl','OmniSurfacePresets.mdl','OmniSurface/OmniSurfaceBase.mdl'}


def references(path):
    if path.suffix in ('.usd','.usda','.usdc'):
        from pxr import Sdf,UsdUtils
        if Sdf.Layer.FindOrOpen(str(path)) is None:
            raise RuntimeError('Invalid USD: '+str(path))
        return sorted({ref for group in UsdUtils.ExtractExternalReferences(str(path)) for ref in group})
    if path.suffix=='.mdl':
        return re.findall(r'texture_[23]d\(\s*"([^"\n]+)"',path.read_text())
    return []


def row(path,root,**extra):
    return {'path':str(path.relative_to(root)),'bytes':path.stat().st_size,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),**extra}


def native_closure():
    root=(ROOT/'checkout').resolve()
    queue=[root/rel for rel in NATIVE_SEEDS]
    seen=set();rows=[];edges=[]
    while queue:
        path=queue.pop(0).resolve()
        if path in seen:continue
        seen.add(path)
        if not path.is_relative_to(root):raise RuntimeError('Native dependency outside checkout: '+str(path))
        if not path.is_file():raise FileNotFoundError('Missing native USD dependency: '+str(path))
        if path.read_bytes().startswith(b'version https://git-lfs.github.com/spec'):
            raise RuntimeError('Native asset is only an LFS pointer: '+str(path))
        rows.append(row(path,root))
        for ref in references(path):
            if ref in BUILTIN_MDL:
                edges.append({'source':str(path.relative_to(root)),'reference':ref,'kind':'Isaac builtin MDL'})
                continue
            if '://' in ref:raise RuntimeError('Unexpected remote dependency in native USD: '+ref)
            child=(path.parent/ref).resolve()
            if not child.is_relative_to(root):raise RuntimeError('USD dependency escapes checkout: '+ref)
            edges.append({'source':str(path.relative_to(root)),'reference':ref,
                          'resolved':str(child.relative_to(root)),'kind':'local original asset'})
            queue.append(child)
    return sorted(rows,key=lambda r:r['path']),edges


def official_closure(check):
    root=(ROOT/'assets').resolve();queue=list(OFFICIAL_SEEDS);seen=set();rows=[]
    while queue:
        rel=queue.pop(0)
        if rel in seen:continue
        seen.add(rel)
        path=(root/rel).resolve()
        if not path.is_relative_to(root):raise RuntimeError('Official resource escapes asset root')
        if not path.exists():
            if check:raise FileNotFoundError(path)
            path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_suffix(path.suffix+'.partial')
            subprocess.run(['curl','-fL','--retry','3','--connect-timeout','15','--max-time','120',
                            BASE+rel,'-o',str(temp)],check=True)
            temp.replace(path)
        rows.append(row(path,root,url=BASE+rel))
        for ref in references(path):
            if ref in BUILTIN_MDL:continue
            url=urljoin(BASE+rel,ref)
            if not url.startswith(BASE):raise RuntimeError('Unexpected official dependency: '+url)
            queue.append(url[len(BASE):])
    return sorted(rows,key=lambda r:r['path'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--update-lock',action='store_true')
    args=parser.parse_args()
    native,edges=native_closure()
    official=official_closure(args.check)
    result={'native_seeds':NATIVE_SEEDS,'files':native,'native_reference_edges':edges,
            'official':official,'unresolved_dependencies':[],
            'note':'Walked USD sublayers/references/payloads and MDL textures, not only top-level hashes. Built-in MDL resolves from original Isaac4.5 installation.'}
    lock=ROOT/'asset-manifest.json'
    if args.update_lock:
        lock.write_text(json.dumps(result,indent=2)+'\n')
    else:
        expected=json.loads(lock.read_text())
        for group in ('files','official'):
            if expected[group]!=result[group]:raise RuntimeError('Asset closure changed: '+group)
        if expected['native_reference_edges']!=edges:raise RuntimeError('Native USD references changed')
    print(f"Verified original native USD closure: {len(native)} files; official scene resources: {len(official)} files; unresolved: 0")


if __name__=='__main__':main()
