"""Stage a standalone HF asset tree; never publish or include restricted BEHAVIOR data."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SECRET_NAMES={'omnigibson.key','auth.json','.netrc','.env','credentials','credentials.json'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'upload-hf');a=p.parse_args()
    out=a.output.resolve()
    if out.exists() and any(out.iterdir()):raise RuntimeError('Output must be new or empty; refusing to merge old release')
    if out==ROOT or ROOT.is_relative_to(out) or out.is_relative_to(ROOT/'Assets'):raise ValueError('Unsafe output')
    out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT))
    from environment.evaluation.rollout_catalog import catalog
    selected=set(catalog())|{'_shared'}
    supplement={r['path']:r for r in json.loads((ROOT/'environment/datasets/runtime-supplement.json').read_text())}
    rows=[]
    for bench in sorted(selected-{'behavior_1k'}):
        source=ROOT/'Assets'/bench
        if not source.is_dir():raise FileNotFoundError(source)
        print('Staging '+bench,flush=True)
        for path in sorted(source.rglob('*')):
            if not path.is_file():continue
            if path.is_symlink():raise RuntimeError('Asset symlink requires explicit materialization: '+str(path))
            if path.suffix.lower() in {'.key','.pem','.p12','.pfx'} or path.name in SECRET_NAMES or path.name.startswith('.env.'):continue
            if set(path.relative_to(source).parts)&{'.git','__pycache__','.cache','logs'}:continue
            rel=Path('Assets')/bench/path.relative_to(source)
            entry=supplement.get(str(rel))
            if entry:
                from scripts.restore_runtime_supplement import valid
                if not valid(path,entry):raise ValueError('Supplement checksum mismatch: '+str(rel))
                if entry.get('external'):continue
            dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True)
            # Reflink if supported; otherwise independent copy, never mutable hard links.
            subprocess.run(['cp','--reflink=auto','--preserve=mode,timestamps',str(path),str(dst)],check=True)
            h=hashlib.sha256()
            with dst.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            rows.append({'path':str(rel),'bytes':dst.stat().st_size,'sha256':h.hexdigest()})
    for row in rows:
        if row['path'] in supplement:row.update(supplement[row['path']])
    rows.extend(row for row in supplement.values() if row.get('external'))
    if not set(supplement).issubset(row['path'] for row in rows):raise ValueError('Missing runtime supplement')
    shutil.copy2(ROOT/'scripts/restore_runtime_supplement.py',out)
    shutil.copy2(ROOT/'environment/datasets/runtime-supplement.json',out)
    data={'version':1,'files':rows,'excluded':['behavior_1k (official non-redistribution terms)','aerial_balance (removed task)','credentials and runtime state'],
          'publication_notice':'Preserve source licenses. This local staging operation does not grant redistribution rights or publish data.'}
    (out/'asset-manifest.json').write_text(json.dumps(data,indent=2)+'\n')
    shutil.copy2(ROOT/'environment/datasets/asset-layout.json',out/'asset-layout.json')
    shutil.copy2(ROOT/'docs/HF_ASSET_CARD.md', out/'README.md')
    print(json.dumps({'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'output':str(out)}),flush=True)
if __name__=='__main__':main()
