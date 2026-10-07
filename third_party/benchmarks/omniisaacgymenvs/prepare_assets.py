"""Fetch only the original Isaac4.0 ANYmal USD reference closure; no conversions."""
import hashlib,json,subprocess
from pathlib import Path
from urllib.parse import urljoin
from pxr import Sdf,UsdUtils
ROOT=Path(__file__).resolve().parent
BASE='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0/'
def main():
    dest=ROOT/'assets'; queue=['Isaac/Robots/ANYbotics/anymal_instanceable.usd']; seen=set(); result=[]
    manifest=ROOT/'asset-manifest.json'
    expected={r['path']:r['sha256'] for r in json.loads(manifest.read_text())} if manifest.exists() else {}
    while queue:
        rel=queue.pop(0)
        if rel in seen: continue
        seen.add(rel); p=(dest/rel).resolve()
        if not p.is_relative_to(dest.resolve()):raise ValueError('Asset path escapes root')
        url=BASE+rel;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            temp=p.with_suffix(p.suffix+'.partial')
            subprocess.run(['curl','-fL','--retry','2','--retry-all-errors','--max-time','90',url,'-o',str(temp)],check=True)
            temp.replace(p)
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if expected and expected.get(rel)!=digest:raise ValueError('Pinned asset mismatch '+rel)
        if p.suffix.lower() in ('.usd','.usda','.usdc'):
            if Sdf.Layer.FindOrOpen(str(p)) is None:raise ValueError('Invalid USD '+rel)
            for ref in sorted({r for group in UsdUtils.ExtractExternalReferences(str(p)) for r in group}):
                if not ref or (ref.endswith('.mdl') and '/' not in ref):continue
                target=urljoin(url,ref)
                if not target.startswith(BASE):raise ValueError('Unexpected reference '+target)
                queue.append(target[len(BASE):])
        result.append({'path':rel,'sha256':digest,'bytes':p.stat().st_size,'url':url})
    manifest.write_text(json.dumps(sorted(result,key=lambda r:r['path']),indent=2)+'\n')
    print('Asset closure verified:',len(result),'files')
if __name__=='__main__':main()
