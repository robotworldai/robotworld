"""Fetch an official Git-LFS USD dependency closure, leaving original asset bytes intact."""
import argparse,hashlib,json,re,shutil,subprocess
from pathlib import Path

def main():
 from pxr import Sdf,UsdUtils
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',action='append',required=True);a=p.parse_args()
 benchmark_root=Path(__file__).resolve().parent;repo=benchmark_root/'checkout';dest=benchmark_root/'assets'
 tracked=subprocess.check_output(['git','-C',str(repo),'ls-files','assets'],text=True).splitlines()
 roots=list(a.scene)+['franka_robotiq_2f_85_flattened.usd','franka_table.usd','home_office.exr','empty_warehouse.hdr','billiard_hall.hdr','brown_photostudio.hdr']
 queue=[]
 for name in roots:
  matches=[x for x in tracked if x.endswith('/'+name)]
  if len(matches)!=1:raise ValueError(f'Ambiguous/missing root {name}: {matches}')
  queue+=matches
 seen=set();external=[]
 while queue:
  batch=sorted(set(queue)-seen);queue=[]
  if not batch:break
  subprocess.run(['git','-C',str(repo),'lfs','pull','--include',','.join(batch),'--exclude',''],check=True)
  for name in batch:
   src=repo/name
   if src.read_bytes()[:80].startswith(b'version https://git-lfs'):raise RuntimeError('Unresolved LFS pointer '+name)
   dst=dest/Path(name).relative_to('assets');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);seen.add(name)
   if src.suffix.lower()=='.mdl':
    refs=set(re.findall(r'texture_\w+\s*\(\s*"([^"]+)"',src.read_text()))
   elif src.suffix.lower() in ('.usd','.usda','.usdc'):
    layer=Sdf.Layer.FindOrOpen(str(src))
    if layer is None:raise RuntimeError('Cannot read USD '+name)
    refs={ref for group in UsdUtils.ExtractExternalReferences(str(src)) for ref in group}
   else:continue
   for ref in refs:
    if not ref:continue
    if '://' in ref:external.append({'source':name,'reference':ref});continue
    candidate=(src.parent/ref).resolve()
    if candidate.is_relative_to(repo.resolve()):
     rel=str(candidate.relative_to(repo.resolve()))
     if rel in tracked:queue.append(rel)
     else:external.append({'source':name,'reference':ref})
    else:external.append({'source':name,'reference':ref})
 manifest={'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
           'scenes':a.scene,'files':[{'path':x.removeprefix('assets/'),'sha256':hashlib.sha256((repo/x).read_bytes()).hexdigest(),'bytes':(repo/x).stat().st_size} for x in sorted(seen)],'unresolved_external':external}
 (benchmark_root/'assets.local.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Copied',len(seen),'files; external references',len(external))
if __name__=='__main__':main()
