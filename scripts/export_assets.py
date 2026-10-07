"""Stage a standalone HF asset tree; never publish or include restricted BEHAVIOR data."""
import argparse,hashlib,json,os,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SECRET_NAMES={'omnigibson.key','auth.json','.netrc','.env','credentials','credentials.json'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'upload-hf');a=p.parse_args()
    out=a.output.resolve()
    if out.exists() and any(out.iterdir()):raise RuntimeError('Output must be new or empty; refusing to merge old release')
    if out==ROOT or ROOT.is_relative_to(out) or out.is_relative_to(ROOT/'Assets'):raise ValueError('Unsafe output')
    out.mkdir(parents=True,exist_ok=True)
    from environment.evaluation.rollout_catalog import catalog
    selected=set(catalog())|{'_shared'}
    rows=[]
    for bench in sorted(selected-{'behavior_1k'}):
        source=ROOT/'Assets'/bench
        if not source.is_dir():raise FileNotFoundError(source)
        print('Staging '+bench,flush=True)
        for path in sorted(source.rglob('*')):
            if not path.is_file():continue
            if path.is_symlink():raise RuntimeError('Asset symlink requires explicit materialization: '+str(path))
            if path.name in SECRET_NAMES or path.name.startswith('.env.'):continue
            if set(path.relative_to(source).parts)&{'.git','__pycache__','.cache','logs'}:continue
            rel=Path('Assets')/bench/path.relative_to(source);dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True)
            # Reflink if supported; otherwise independent copy, never mutable hard links.
            subprocess.run(['cp','--reflink=auto','--preserve=mode,timestamps',str(path),str(dst)],check=True)
            h=hashlib.sha256()
            with dst.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            rows.append({'path':str(rel),'bytes':dst.stat().st_size,'sha256':h.hexdigest()})
    data={'version':1,'files':rows,'excluded':['behavior_1k (official non-redistribution terms)','aerial_balance (removed task)','credentials and runtime state'],
          'publication_notice':'Preserve source licenses. This local staging operation does not grant redistribution rights or publish data.'}
    (out/'asset-manifest.json').write_text(json.dumps(data,indent=2)+'\n')
    shutil.copy2(ROOT/'environment/datasets/asset-layout.json',out/'asset-layout.json')
    (out/'README.md').write_text('''---
license: other
license_name: per-upstream-asset-terms
license_link: https://github.com/REPLACE_ORG/REPLACE_REPO/blob/main/docs/ASSETS.md
---
# RobotWorld selected assets

Download using the paired code release's `scripts/download_assets.py`. Keep `Assets/` beneath the repository root, then run `python scripts/restore_assets.py --apply --bench BENCH --include-shared`.

This is a local handoff staging directory, not a published dataset. Each third-party asset retains its upstream license; there is no blanket license grant. Review original terms before public redistribution. BEHAVIOR-1K assets and keys are excluded and must be obtained by each user through the official agreement. Removed aerial_balance/T04 data is excluded. No model evaluation results, credentials, Docker images or Python environments are included.

`asset-manifest.json` records every included file's SHA-256 and size. Edit the metadata/license links and the code release's asset-source URL before publishing. Some upstream missing assets remain missing; packaging does not imply every task is physically validated.
''')
    print(json.dumps({'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'output':str(out)}),flush=True)
if __name__=='__main__':main()
