"""Restore pinned independent checkouts, preserving bundled source backups."""
import argparse, datetime, json, os, shutil, subprocess, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 rows=json.loads((ROOT/'sources.lock.json').read_text())
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('names',nargs='*');p.add_argument('--all',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--cache-dir',type=Path,help='Snapshot directory on the same filesystem as source checkouts');a=p.parse_args()
 known={r['id'] for r in rows};chosen=known if a.all else set(a.names)
 if not chosen or chosen-known:p.error('Select sources from: '+', '.join(sorted(known)))
 def git(*args,cwd=None):return subprocess.check_output(['git',*args],cwd=cwd,env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1'},text=True).strip()
 for r in rows:
  if r['id'] not in chosen:continue
  dest=ROOT/r['path'];print(r['id'],r['url'],r['commit'],flush=True)
  if not dest.resolve().is_relative_to(ROOT):raise RuntimeError('Destination escapes repository')
  if a.dry_run:continue
  def restore_submodules(checkout):
   if (checkout/'.gitmodules').exists():git('submodule','update','--init','--recursive',cwd=checkout)
  if (dest/'.git').exists():
   if git('rev-parse','HEAD',cwd=dest)!=r['commit'] or git('status','--porcelain',cwd=dest):raise RuntimeError('Existing checkout differs or is dirty: '+str(dest))
   restore_submodules(dest)
   continue
  cache=a.cache_dir or ROOT/'var/source-snapshots';cache.mkdir(parents=True,exist_ok=True)
  if cache.stat().st_dev!=dest.parent.stat().st_dev:raise RuntimeError('Snapshot cache must share the checkout filesystem; use --cache-dir')
  temp=Path(tempfile.mkdtemp(prefix=r['id']+'-',dir=cache))
  try:
   git('init',str(temp));git('remote','add','origin',r['url'],cwd=temp);git('fetch','--depth=1','origin',r['commit'],cwd=temp);git('checkout','--detach',r['commit'],cwd=temp)
   if git('rev-parse','HEAD',cwd=temp)!=r['commit']:raise RuntimeError('Revision mismatch')
   restore_submodules(temp)
   backup=None
   if dest.exists():
    backup=cache/(r['id']+'-bundled-'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f'));dest.rename(backup)
   try:dest.parent.mkdir(parents=True,exist_ok=True);temp.rename(dest)
   except BaseException:
    if backup is not None:backup.rename(dest)
    raise
  finally:
   if temp.exists():shutil.rmtree(temp)
  print('Ready:',dest,'(LFS assets remain separate)',flush=True)
if __name__=='__main__':main()
