"""Index upstream-bundled assets without dirtying immutable checkouts.
Materialized assets use hard links in Assets: one physical copy, original pathname
and bytes remain unchanged. Code exports can exclude the listed upstream paths.
"""
import argparse,hashlib,json,os
from pathlib import Path
WORLD=Path(__file__).resolve().parents[2]
EXT={'.urdf','.srdf','.sdf','.pkl','.ckpt','.safetensors','.mtlx','.tif','.tiff','.usd','.usda','.usdc','.usdz','.stl','.obj','.dae','.ply','.glb','.gltf','.hdr','.exr','.mdl','.mtl','.npy','.npz','.onnx','.pt','.pth','.h5','.hdf5','.png','.jpg','.jpeg','.tga','.bmp','.dds','.ktx'}
SKIP={'.git','__pycache__','.cache','.venv','node_modules','outputs','logs','wandb'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');p.add_argument('--manifest',type=Path,default=WORLD/'environment/datasets/embedded-assets.json');a=p.parse_args();rows=[];unmaterialized=[]
 roots=[(x.parent.name,x,'checkout') for x in (WORLD/'third_party/benchmarks').glob('*/checkout')]
 for root in sorted((WORLD/'third_party/dependencies').glob('*/checkout')):
  name=root.parent.name
  roots.append(('robocasa' if name=='robosuite' else '_shared',root,'robosuite' if name=='robosuite' else 'dependencies/'+name))
 for root in sorted((WORLD/'third_party/benchmarks').glob('*/isaaclab*')):
  if root.is_dir():roots.append((root.parent.name,root,root.name))
 for bench,root,component in roots:
  for source in sorted(root.rglob('*')):
   relative=source.relative_to(root)
   if set(relative.parts)&SKIP or not source.is_file() or source.is_symlink():continue
   is_xml=source.suffix=='.xml' and ('models' in relative.parts or 'assets' in relative.parts or 'sim2sim_mujoco' in relative.parts or 'g1_xml' in relative.parts)
   if source.suffix.lower() not in EXT and not is_xml:continue
   # Documentation illustrations are source documentation, not simulation resources.
   if set(relative.parts)&{'docs','media','_static','figures'}:continue
   with source.open('rb') as f:
    first=f.read(128)
    if first.startswith(b'version https://git-lfs.github.com/spec/v1'):
     unmaterialized.append(str(source.relative_to(WORLD)));continue
   target=WORLD/'Assets'/bench/'upstream'/component/relative
   if a.apply:
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
     if not os.path.samefile(source,target):raise RuntimeError('Existing non-identical destination: '+str(target))
    else:os.link(source,target)
   h=hashlib.sha256()
   with source.open('rb') as f:
    while block:=f.read(4*1024*1024):h.update(block)
   rows.append({'benchmark':bench,'source':str(source.relative_to(WORLD)),'asset':str(target.relative_to(WORLD)),'bytes':source.stat().st_size,'sha256':h.hexdigest(),'redistribution':'prohibited' if bench=='behavior_1k' else 'requires source license compliance'})
 a.manifest.parent.mkdir(parents=True,exist_ok=True);a.manifest.write_text(json.dumps({'mode':'hardlinks_created' if a.apply else 'plan','upstream_modified':False,'assets_immutable':True,'files':rows,'unmaterialized_lfs_not_downloaded':unmaterialized},indent=2));print(json.dumps({'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'unmaterialized_lfs':len(unmaterialized),'applied':a.apply}))
if __name__=='__main__':main()
