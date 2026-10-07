"""Download only the original Isaac 2022.2.0 Franka reference closure, without edits."""
import hashlib, json, re, subprocess
from pathlib import Path
from urllib.parse import urljoin, urlparse
from pxr import Sdf, UsdUtils
ROOT=Path(__file__).resolve().parent
BASE='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2022.2.0/'
def main():
    expected={r['path']:r['sha256'] for r in json.loads((ROOT/'asset-manifest.json').read_text())}
    dest=ROOT/'assets'; queue=['Isaac/Robots/Franka/franka_instanceable.usd','Isaac/Environments/Grid/default_environment.usd']; seen=set(); manifest=[]
    while queue:
        rel=queue.pop(0)
        if rel in seen:continue
        seen.add(rel); p=(dest/rel).resolve()
        if not p.is_relative_to(dest.resolve()):raise ValueError('Asset path escapes destination')
        url=BASE+rel;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            print('Download',url,flush=True)
            temp=p.with_suffix(p.suffix+'.partial')
            subprocess.run(['curl','-fL','--retry','3','--retry-all-errors','--max-time','120',url,'-o',str(temp)],check=True)
            temp.replace(p)
        if p.suffix in ('.usd','.usda','.usdc'):
            layer=Sdf.Layer.FindOrOpen(str(p))
            if layer is None:raise ValueError('Unreadable USD '+str(p))
            refs={ref for group in UsdUtils.ExtractExternalReferences(str(p)) for ref in group}
            for ref in sorted(refs):
                if not ref or (ref.endswith('.mdl') and '/' not in ref):continue
                linked=urljoin(url,ref)
                if not linked.startswith(BASE):
                    # Built-in MDL modules resolve from Isaac installation, not remote scene geometry.
                    if ref.endswith('.mdl') and '/' not in ref:continue
                    raise ValueError('Unexpected external asset '+linked)
                queue.append(linked[len(BASE):])
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if rel not in expected or digest!=expected[rel]:raise ValueError('Asset does not match pinned SHA256: '+rel)
        manifest.append({'path':rel,'url':url,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (dest/'manifest.json').write_text(json.dumps(sorted(manifest,key=lambda x:x['path']),indent=2))
    print('Ready:',len(manifest),'files',sum(m['bytes'] for m in manifest),'bytes')
if __name__=='__main__':main()
