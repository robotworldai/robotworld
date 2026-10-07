"""Export GitHub-ready source files, excluding assets, nested Git and local state.
No publication or change to the live upstream checkouts. Restore source Git
metadata using fetch_sources.py before provenance-checked runtime launches.
"""
import argparse,hashlib,json,os,shutil,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ROOT_DIRS={'codex','third_party','environment','scripts','tests','docs'}
SKIP={'.git','.cache','__pycache__','.pytest_cache','node_modules','target','dist','build','build-context','.venv','venv','outputs','var','wandb','logs','tensorboard','archive'}
PRIVATE_DIAGNOSTICS={'environment/evaluation/world_success/reference.py','environment/evaluation/world_success/reference_vehicle.py','scripts/report_reachability.py','scripts/report_world_success_validation.py','third_party/benchmarks/wheeledlab/robotworld/precision_reference.py'}
BIN={'.pkl','.ckpt','.safetensors','.usd','.usda','.usdc','.usdz','.stl','.obj','.dae','.ply','.glb','.gltf','.onnx','.pt','.pth','.npz','.npy','.h5','.hdf5','.tar','.gz','.zip','.7z','.mp4','.avi','.so','.a','.o','.pyc'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args();out=a.output.resolve()
 if out==ROOT or ROOT.is_relative_to(out):raise ValueError('Output must not contain the development root')
 embedded=ROOT/'environment/datasets/embedded-assets.json';metadata=json.loads(embedded.read_text()) if embedded.exists() else {}
 excluded={x['source'] for x in metadata.get('files',[])}|set(metadata.get('unmaterialized_lfs_not_downloaded',[]))
 candidates=[]
 for directory in sorted(ROOT_DIRS):
  for base,dirs,files in os.walk(ROOT/directory,followlinks=False):
   path=Path(base);rel=path.relative_to(ROOT)
   dirs[:]=[x for x in dirs if x not in SKIP and not (path/x).is_symlink() and not (str(rel)=='environment/configs' and x=='local')]
   # Codex assets include compiled-in prompts, tool grammars and UI resources.
   if rel.parts[0]!='codex' and 'assets' in rel.parts and 'checkout' not in rel.parts:dirs[:]=[];continue
   if any(x in rel.parts for x in ('dex2bench_dataset','anchors','shared-assets')):dirs[:]=[];continue
   for name in files:
    f=path/name;r=str(f.relative_to(ROOT))
    if (f.suffix.lower() in {'.png','.jpg','.jpeg','.gif','.webp','.pptx','.pdf'} and set(f.relative_to(ROOT).parts)&{'docs','media','_static','figures'}) or name in {'assets.local.json','image-inspect.json','provenance.json'}:continue
    if r in PRIVATE_DIAGNOSTICS:continue
    if f.is_symlink() or r in excluded or f.suffix.lower() in BIN|{'.log','.webm','.mov','.mkv'} or name.startswith('events.out.tfevents.') or name in ('.git','auth.json','omnigibson.key','.env'):continue
    if name.startswith('.env.') and name!='.env.example':continue
    candidates.append(f)
 candidates += [p for p in ROOT.iterdir() if p.is_file() and (p.suffix=='.md' or p.name in ('pyproject.toml','sources.lock.json','.gitignore'))]
 candidates += [ROOT/'robotworld-scoring-guide.html']
 # Handoff diagnostics only; historical run reports are intentionally omitted.
 for f in (ROOT/'reports/handoff').glob('*'):
  if f.is_file() and f.suffix in ('.md','.json'):candidates.append(f)
 if a.dry_run:
  print(json.dumps({'files':len(candidates),'bytes':sum(x.stat().st_size for x in candidates),'output':str(out),'largest':[(str(x.relative_to(ROOT)),x.stat().st_size) for x in sorted(candidates,key=lambda p:p.stat().st_size,reverse=True)[:12]]}));return
 if out.exists() and any(out.iterdir()):raise ValueError('Output must be new or empty; refusing to merge stale code')
 out.mkdir(parents=True,exist_ok=True);records=[]
 for source in sorted(set(candidates)):
  rel=source.relative_to(ROOT);target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
  if str(rel) in ('environment/runtime/native_project_launch.py','third_party/benchmarks/wheeledlab/docker/run.py'):
   target.write_text(target.read_text().replace("['probe','zero','codex','reference']", "['probe','zero','codex']"))
  if str(rel)=='environment/integrations/native_project_eval.py':
   text=target.read_text().replace("['probe','zero','codex','reference']", "['probe','zero','codex']")
   text=re.sub(r"            reference=None\n            if a.mode=='reference':\n.*?(?=            for _ in range)", '', text, flags=re.S)
   text=text.replace('                if reference is not None:action=reference()\n','')
   text=re.sub(r"        if a.mode=='reference':\n.*?(?=        result.update)", '', text, flags=re.S)
   target.write_text(text)
  if str(rel)=='environment/integrations/wheeledlab_eval.py':
   text=target.read_text().replace("['probe','zero','codex','reference']", "['probe','zero','codex']")
   text=re.sub(r"            reference=None\n            if a.mode=='reference':\n.*?(?=            for _ in range)", '', text, flags=re.S)
   text=re.sub(r"                action=.*?if reference else \[0\.,0\.\]", '                action=[0.,0.]', text)
   text=re.sub(r"        if a.mode=='reference':result.update[^\n]*\n", '', text)
   target.write_text(text)
  if str(rel) in ('environment/validation/commands.py','environment/validation/remaining_campaign.py'):
   # The release contains no privileged reference controller: use no-action smoke validation.
   target.write_text(target.read_text().replace("'reference' if k in ('rw-twin-beam','rw-reverse-bay','rw-parallel-park') else 'zero'", "'zero'"))
  if str(rel)=='.gitignore':
   lines=target.read_text().splitlines()
   target.write_text('\n'.join(line for line in lines if not (line.startswith('/third_party/') and (line.endswith('checkout/') or line.endswith('isaaclab211/') or line=='/third_party/benchmarks/RoboDojo/')))+'\n')
  if str(rel)=='robotworld-scoring-guide.html':
   page=target.read_text()
   page=re.sub(r'href="(?:var/|reports/validation/|docs/scoring/archive/)[^"]*"', 'href="docs/HANDOFF.md"', page)
   page=page.replace('<body>', '<body><p style="padding:16px;text-align:center;background:#fff1d0">This source export does not include historical videos. Historical evidence links lead to the handoff notes. New results are stored in outputs/ after evaluation.</p>')
   target.write_text(page)
  records.append({'path':str(rel),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size})
 (out/'code-export.json').write_text(json.dumps({'assets_included':False,'nested_git_included':False,'files':records},indent=2))
 (out/'RESTORE.md').write_text('# Restore the handoff\n\nSee README.md and docs/HANDOFF.md. Run setup_handoff.sh sources before asset restoration, then configure local source Codex and Docker. No assets, runtime environments, credentials or results are included.\n')
 print(json.dumps({'files':len(records),'bytes':sum(x['bytes'] for x in records),'output':str(out)}))
if __name__=='__main__':main()
