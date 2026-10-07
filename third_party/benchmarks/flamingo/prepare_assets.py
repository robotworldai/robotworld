"""Verify pinned upstream files and explicit robot assets; never mutate checkout."""
from pathlib import Path
import argparse,hashlib,json,subprocess,zipfile
ROOT=Path(__file__).resolve().parent
ASSETS=['lab/flamingo/assets/data/Robots/Flamingo/flamingo_rev01_5_2/flamingo_rev01_5_2_merge_joints.zip','lab/flamingo/assets/data/ActuatorNets/Flamingo/kan/wheel/symbolic_formula.txt']

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

def extract():
    archive=ROOT/'checkout'/ASSETS[0]
    target=ROOT/'assets/Robots/Flamingo/flamingo_rev01_5_2'
    target.mkdir(parents=True,exist_ok=True)
    files=[]
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            path=(target/entry.filename).resolve()
            if not path.is_relative_to(target.resolve()):raise RuntimeError('Archive escapes destination')
            if entry.is_dir():path.mkdir(parents=True,exist_ok=True);continue
            blob=z.read(entry);path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists() and path.read_bytes()!=blob:raise RuntimeError('Existing extracted asset changed: '+str(path))
            if not path.exists():path.write_bytes(blob)
            files.append({'path':str(path.relative_to(ROOT)),'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()})
    return files

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write-manifest',action='store_true');a=p.parse_args()
    result=verify();result["extracted_files"]=extract();result["extraction"]= "Unmodified ZIP members outside read-only checkout"
    if a.write_manifest:(ROOT/'asset-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'files':len(result['files']),'commit':result['commit']},indent=2))
if __name__=='__main__':main()
