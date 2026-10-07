"""Record upstream hashes/cleanliness and image identities without reading secrets."""
import argparse,hashlib,json,subprocess
from pathlib import Path
WORLD=Path(__file__).resolve().parents[2]
def command(args):
 r=subprocess.run(args,capture_output=True,text=True)
 return r.stdout.strip() if r.returncode==0 else None
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
 roots=[WORLD/'codex',WORLD/'third_party/benchmarks/RoboDojo',*sorted((WORLD/'third_party/benchmarks').glob('*/checkout')),*sorted((WORLD/'third_party/dependencies').glob('*/checkout'))]
 # Some projects pin their own IsaacLab fork beside checkout/.
 roots+=sorted(p for p in (WORLD/'third_party/benchmarks').glob('*/*') if p.is_dir() and (p/'.git').exists() and p not in roots)
 for root in roots:
  rev=command(['git','-C',str(root),'rev-parse','HEAD']);status=command(['git','-C',str(root),'status','--porcelain'])
  extra={}
  if not rev and root.is_dir():
   h=hashlib.sha256();files=0
   for f in sorted(root.rglob('*')):
    if f.is_file() and '__pycache__' not in f.parts and not f.is_symlink():
     h.update(str(f.relative_to(root)).encode()+b'\0');h.update(f.read_bytes());files+=1
   extra={'identity_kind':'source extraction without Git metadata','tree_sha256':h.hexdigest(),'files':files}
  rows.append({**extra,'path':str(root.relative_to(WORLD)),'commit':rev,'source_available':root.is_dir(),'clean':status=='' if rev else None,'changes':status.splitlines() if status else []})
 images=[]
 raw=command(['docker','images','--format','{{.Repository}}:{{.Tag}}']) or ''
 for name in sorted(set(n for n in raw.splitlines() if n.startswith(('world/','stanfordvl/behavior:')))):
  data=command(['docker','image','inspect',name,'--format','{{json .Id}}'])
  images.append({'name':name,'image_id':json.loads(data) if data else None})
 result={'upstream_sources':rows,'images':images,'all_available_upstreams_clean':all(r['clean'] for r in rows if r.get('commit')),'note':'Image identity is recorded, not a claim of official runtime or physics equivalence. Credentials and environment variables are not read.'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps({'repositories':len(rows),'clean':sum(r['clean'] is True for r in rows),'images':len(images)}))
 if not result['all_available_upstreams_clean']:raise SystemExit('Upstream checkout modifications detected; inspect provenance report')
if __name__=='__main__':main()
