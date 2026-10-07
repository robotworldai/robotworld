"""Fetch official 4.5 ground/goal-marker resources; verify bundled vehicle assets."""
import hashlib,json,subprocess
from pathlib import Path

def main():
    root=Path(__file__).resolve().parent;manifest=json.loads((root/'asset-manifest.json').read_text())
    for row in manifest['files']:
        p=root/'checkout/source/wheeledlab_assets/data'/row['path']
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Bundled asset missing/changed: '+str(p))
    base='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/'
    paths=[r['path'] for r in manifest['official_ground']]
    entries=[]
    for rel in paths:
        path=root/'assets'/rel;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            temp=path.with_suffix(path.suffix+'.part')
            subprocess.run(['curl','-fL','--retry','3','--connect-timeout','15','--max-time','120',base+rel,'-o',str(temp)],check=True)
            temp.replace(path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        pinned=next((r for r in manifest.get('official_ground',[]) if r['path']==rel),None)
        if pinned and pinned['sha256']!=digest:raise RuntimeError('Official ground checksum mismatch: '+rel)
        entries.append({'path':rel,'url':base+rel,'bytes':path.stat().st_size,'sha256':digest})
    (root/'assets/manifest.json').write_text(json.dumps(entries,indent=2)+'\n')
    print(json.dumps({'bundled_usd':len(manifest['files']),'downloaded_ground_files':len(entries),'ground_bytes':sum(r['bytes'] for r in entries),'missing_upstream_texture':manifest['unresolved_upstream_texture']},indent=2))

if __name__=='__main__':main()
