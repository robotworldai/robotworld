"""Restore asset paths from a populated Assets directory; no network or publishing.
Hash-checks embedded resources before linking them into pinned source checkouts.
"""
import argparse,hashlib,json,os,errno,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
def link_atomically(target,source):
 source.parent.mkdir(parents=True,exist_ok=True)
 temporary=source.with_name(source.name+f'.world-restore-{os.getpid()}')
 mode=(source.stat().st_mode & 0o777) if source.exists() else (target.stat().st_mode & 0o777)
 try:
  if mode != target.stat().st_mode & 0o777:
   # Do not chmod a shared asset inode merely to preserve a checkout's mode.
   shutil.copy2(target,temporary);temporary.chmod(mode)
  else:
   try:os.link(target,temporary)
   except OSError as e:
    if e.errno!=errno.EXDEV:raise
    shutil.copy2(target,temporary)
  os.replace(temporary,source)
 finally:
  if temporary.exists():temporary.unlink()
def main():
 p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');p.add_argument('--bench',help='Only restore one benchmark; default all');p.add_argument('--include-shared',action='store_true',help='With --bench, also restore shared dependency resources');a=p.parse_args();rows=[]
 layout=json.loads((ROOT/'environment/datasets/asset-layout.json').read_text())
 for item in layout['external']:
  source=ROOT/item['source'];target=ROOT/item['target'];bench=Path(item['target']).parts[1]
  if a.bench and a.bench!=bench:continue
  if not target.is_dir():rows.append({'path':item['target'],'status':'missing'});continue
  if source.is_symlink() and source.resolve()==target.resolve():rows.append({'path':item['source'],'status':'ready'});continue
  if source.exists():rows.append({'path':item['source'],'status':'existing_legacy_directory_requires_migration'});continue
  if source.is_symlink():
   rows.append({'path':item['source'],'status':'dangling_or_different_symlink'})
   continue
  if a.apply:source.parent.mkdir(parents=True,exist_ok=True);source.symlink_to(target.resolve(),target_is_directory=True)
  rows.append({'path':item['source'],'status':'linked' if a.apply else 'would_link'})
 manifest=json.loads((ROOT/layout['embedded_manifest']).read_text())
 for item in manifest['files']:
  if a.bench and item['benchmark']!=a.bench and not (a.include_shared and item['benchmark']=='_shared'):continue
  source=ROOT/item['source'];target=ROOT/item['asset']
  if not target.is_file():rows.append({'path':item['asset'],'status':'missing'});continue
  if target.stat().st_size!=item['bytes'] or digest(target)!=item['sha256']:raise ValueError('Asset checksum mismatch: '+item['asset'])
  if source.exists():
   if digest(source)==item['sha256']:
    if a.apply and not os.path.samefile(source,target):
     link_atomically(target,source);rows.append({'path':item['source'],'status':'deduplicated_identical'})
    continue
   with source.open('rb') as f:ptr=f.read(128).startswith(b'version https://git-lfs.github.com/spec/v1')
   if not ptr:raise RuntimeError('Refusing to overwrite differing source asset: '+str(source))
  if a.apply:link_atomically(target,source)
  rows.append({'path':item['source'],'status':'restored' if a.apply else 'would_restore'})
 out=ROOT/'var/asset-restore.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps({'applied':a.apply,'files':rows},indent=2))
 from collections import Counter
 print(json.dumps(dict(Counter(r['status'] for r in rows))))
 if any(r['status'] in ('missing','dangling_or_different_symlink') for r in rows):raise SystemExit(1)
if __name__=='__main__':main()
