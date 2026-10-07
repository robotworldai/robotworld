"""Download selected HF assets, verify every SHA-256, and restore adapter paths."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    from environment.evaluation.rollout_catalog import catalog
    cfg=json.loads((ROOT/'environment/datasets/asset-source.json').read_text())
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo-id',default=os.environ.get('WORLD_ASSET_REPO',cfg['repo_id']))
    p.add_argument('--revision',default=cfg['revision']);p.add_argument('--bench',action='append',choices=list(catalog()))
    p.add_argument('--local-source',type=Path,help='Use an existing upload-hf folder offline');p.add_argument('--no-restore',action='store_true');a=p.parse_args()
    benches=set(a.bench or catalog())-{'behavior_1k'}
    if a.local_source:
        source=a.local_source.resolve()
    else:
        if 'REPLACE_' in a.repo_id:p.error('Set environment/datasets/asset-source.json repo_id or --repo-id to the published HF dataset')
        from huggingface_hub import snapshot_download
        source=Path(snapshot_download(repo_id=a.repo_id,repo_type='dataset',revision=a.revision,
            allow_patterns=['asset-manifest.json']+[f'Assets/{b}/**' for b in sorted(benches|{'_shared'})],
            cache_dir=str(ROOT/'var/cache/huggingface')))
    manifest_file=source/'asset-manifest.json'
    if cfg.get('manifest_sha256') and checksum(manifest_file)!=cfg['manifest_sha256']:
        raise ValueError('Asset manifest differs from the paired code release; check dataset revision')
    manifest=json.loads(manifest_file.read_text());count=0
    for row in manifest['files']:
        rel=Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts or len(rel.parts)<3 or rel.parts[0]!='Assets':raise ValueError('Unsafe asset path')
        if rel.parts[1] not in benches|{'_shared'}:continue
        src=source/rel;dst=ROOT/rel
        if not src.is_file() or src.stat().st_size!=row['bytes'] or checksum(src)!=row['sha256']:raise ValueError('Asset checksum mismatch: '+str(rel))
        if dst.exists():
            if checksum(dst)!=row['sha256']:raise ValueError('Existing differing asset; refusing overwrite: '+str(dst))
        else:
            dst.parent.mkdir(parents=True,exist_ok=True)
            tmp=dst.with_name(dst.name+'.download-'+str(os.getpid()))
            try:shutil.copy2(src,tmp);tmp.replace(dst)
            finally:
                if tmp.exists():tmp.unlink()
        count+=1
    failures=[]
    if not a.no_restore:
        for bench in sorted(benches):
            if subprocess.run([sys.executable,str(ROOT/'scripts/restore_assets.py'),'--apply','--bench',bench,'--include-shared']).returncode:failures.append(bench)
    print(json.dumps({'verified_files':count,'restore_failures':failures,'behavior':'Official user-accepted download required; not redistributed'}))
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
