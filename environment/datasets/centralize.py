"""Move external data into Assets, preserving legacy imports with relative links.
No upstream checkout files are changed. apply is explicit; plan is read-only.
"""
import argparse,hashlib,json,os
from pathlib import Path
WORLD=Path(__file__).resolve().parents[2]

def mappings():
 layout=WORLD/'environment/datasets/asset-layout.json'
 if layout.exists():return json.loads(layout.read_text())['external']
 rows=[]
 for source in sorted((WORLD/'third_party/benchmarks').glob('*/assets')):
  rows.append({'source':str(source.relative_to(WORLD)),'target':f'Assets/{source.parent.name}/data'})
 for name in ['behavior_1k','robodojo']:
  source=WORLD/'var/datasets'/name
  if source.exists():rows.append({'source':str(source.relative_to(WORLD)),'target':f'Assets/{name}/'+('data' if name=='behavior_1k' else 'legacy-scenes')})
 for name in ['anchors','dex2bench_dataset','shared-assets']:
  rows.append({'source':'third_party/benchmarks/bench2dex/'+name,'target':'Assets/bench2dex/'+name})
 return rows

def digest_tree(root):
 # Hash names, lengths, contents and links; do not traverse links outside the bundle.
 h=hashlib.sha256();count=size=0
 def raise_walk_error(error):raise error
 paths=[]
 for base,dirs,files in os.walk(root,followlinks=False,onerror=raise_walk_error):
  paths.extend(Path(base)/name for name in dirs+files)
 for p in sorted(paths):
  if p.is_symlink():h.update(('link:'+str(p.relative_to(root))+':'+os.readlink(p)).encode());continue
  if not p.is_file():continue
  h.update(str(p.relative_to(root)).encode()+b'\0');n=p.stat().st_size;h.update(str(n).encode()+b'\0');count+=1;size+=n
  with p.open('rb') as f:
   while block:=f.read(4*1024*1024):h.update(block)
 return {'files':count,'bytes':size,'tree_sha256':h.hexdigest()}

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','apply','verify']);p.add_argument('--bench',help='Limit to one benchmark for a fresh partial installation');p.add_argument('--report',type=Path,default=WORLD/'reports/assets/migration.json');a=p.parse_args();a.report.parent.mkdir(parents=True,exist_ok=True)
 rows=mappings()
 if a.bench:
  rows=[row for row in rows if Path(row['target']).parts[1]==a.bench]
  if not rows:p.error('No external asset mapping for this benchmark; consult embedded-assets.json')
 if a.action=='plan':
  a.report.write_text(json.dumps({'action':'plan','mappings':rows},indent=2));print(a.report);return
 for row in rows:
  source=WORLD/row['source'];target=WORLD/row['target']
  if a.action=='apply':
   if source.is_symlink():
    if source.resolve()!=target.resolve():raise RuntimeError('Unexpected existing link '+str(source))
    if not target.is_dir():raise FileNotFoundError('Canonical asset directory is missing: '+str(target))
    row['status']='already_migrated';row['after']=digest_tree(target)
   else:
    if not source.is_dir():raise FileNotFoundError('Asset source directory is missing: '+str(source))
    if target.exists():raise FileExistsError(target)
    before=digest_tree(source);target.parent.mkdir(parents=True,exist_ok=True)
    source.rename(target)
    try:source.symlink_to(os.path.relpath(target,source.parent),target_is_directory=True)
    except BaseException:target.rename(source);raise
    after=digest_tree(target)
    if before!=after:raise RuntimeError('Asset contents changed during relocation: '+str(target))
    row.update(status='migrated_verified',before=before,after=after)
  else:
   row['linked_to_canonical']=source.is_symlink() and source.resolve()==target.resolve()
   row['exists']=target.is_dir()
  a.report.write_text(json.dumps({'action':a.action,'upstream_sources_changed':False,'mappings':rows},indent=2))
 print(a.report)
 if a.action=='verify' and not all(r['linked_to_canonical'] and r['exists'] for r in rows):raise SystemExit(1)
if __name__=='__main__':main()
