"""Download only the native Go2 USD dependency closure from NVIDIA's official root."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess,urllib.parse
ROOT=Path(__file__).resolve().parent
BASE='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/'
ENTRY='Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd'
LAB_COMMIT='90b79bb2d44feb8d833f260f2bf37da3487180ba'

def verify_checkout(path,revision):
    if subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()!=revision:raise RuntimeError('Wrong pinned source '+str(path))
    if subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip():raise RuntimeError('Dirty pinned source '+str(path))

def prepare(download=False):
    from pxr import Sdf
    lock=json.loads((ROOT/'project.json').read_text());verify_checkout(ROOT/'checkout',lock['commit']);verify_checkout(ROOT/'isaaclab211',LAB_COMMIT)
    pending=[BASE+ENTRY, BASE+'Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr'];seen=set();rows=[];missing=[];engine_builtins=set()
    while pending:
        url=pending.pop()
        if url in seen:continue
        seen.add(url)
        if not url.startswith(BASE):raise RuntimeError('Dependency outside approved official asset root: '+url)
        rel=urllib.parse.unquote(url[len(BASE):]);path=ROOT/'assets'/rel
        if not path.resolve().is_relative_to((ROOT/'assets').resolve()):raise RuntimeError('Escaping asset path')
        if not path.exists() and download:
            path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.part')
            subprocess.run(['curl','-fL','--retry','2','--connect-timeout','15','--max-time','180',url,'-o',str(temp)],check=True);temp.replace(path)
        if not path.exists():missing.append(url);continue
        blob=path.read_bytes();rows.append({'path':rel,'url':url,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()})
        if path.suffix.lower() in ['.usd','.usda','.usdc']:
            layer=Sdf.Layer.FindOrOpen(str(path))
            if layer is None:raise RuntimeError('Invalid USD: '+str(path))
            references=set(re.findall(r'@([^@]+)@',layer.ExportToString()))
            for ref in references:
                if ref in {'OmniPBR.mdl','OmniGlass.mdl'}:
                    engine_builtins.add(ref);continue
                if ref:pending.append(urllib.parse.urljoin(url,ref))
    manifest={'repository':lock['repository'],'commit':lock['commit'],'task_dependency':{'repository':'https://github.com/isaac-sim/IsaacLab.git','tag':'v2.1.1','commit':LAB_COMMIT},'entry':ENTRY,'files':sorted(rows,key=lambda x:x['path']),'missing':missing,'complete':not missing,'engine_builtins':sorted(engine_builtins)}
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--download',action='store_true');p.add_argument('--write-manifest',action='store_true');a=p.parse_args()
    result=prepare(a.download)
    if a.write_manifest:(ROOT/'asset-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'files':len(result['files']),'complete':result['complete'],'missing':result['missing']},indent=2))
    if result['missing']:raise SystemExit(2)
if __name__=='__main__':main()
