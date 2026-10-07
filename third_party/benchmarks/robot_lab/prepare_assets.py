"""Verify pinned upstream files and explicit robot assets; never mutate checkout."""
from pathlib import Path
import argparse,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parent
ASSETS=['source/robot_lab/data/Robots/Unitree/A1']

def verify():
    lock=json.loads((ROOT/'project.json').read_text());src=ROOT/'checkout'
    head=subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()
    if head!=lock['commit']:raise RuntimeError('Wrong checkout revision: '+head)
    if subprocess.check_output(['git','-C',str(src),'status','--porcelain'],text=True).strip():raise RuntimeError('Expected unmodified upstream checkout')
    rows=[]
    for relative in ASSETS:
        path=src/relative
        if not path.exists():raise FileNotFoundError(path)
        paths=sorted(path.rglob('*')) if path.is_dir() else [path]
        for f in paths:
            if not f.is_file():continue
            blob=f.read_bytes()
            if blob.startswith(b'version https://git-lfs.github.com/spec'):raise RuntimeError('Unresolved LFS pointer: '+str(f))
            rows.append({'path':str(f.relative_to(src)),'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()})
    return {'repository':lock['repository'],'commit':head,'files':rows,'upstream_modified':False}

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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write-manifest',action='store_true');p.add_argument('--download',action='store_true');a=p.parse_args()
    result=verify()
    result['native_hdr']=prepare_hdr(a.download)
    if a.write_manifest:(ROOT/'asset-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'files':len(result['files']),'commit':result['commit']},indent=2))
if __name__=='__main__':main()
