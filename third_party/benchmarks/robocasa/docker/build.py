"""Build evaluation image from clean source archives, excluding all local assets."""
import io,subprocess,tarfile,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
WORLD=HERE.parents[3]
context=WORLD/'var/build/docker/robocasa/build-context'
context.mkdir(parents=True,exist_ok=True)
sources={}
for name,root in [('robocasa',HERE.parent/'checkout'),('robosuite',WORLD/'third_party/dependencies/robosuite/checkout')]:
    rev=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(root),'status','--porcelain']):raise RuntimeError(f'Dirty source: {root}')
    raw=subprocess.check_output(['git','-C',str(root),'archive',rev])
    with tarfile.open(fileobj=io.BytesIO(raw)) as ar:ar.extractall(context/name,filter='data')
    sources[name]=rev
(context/'Dockerfile').write_bytes((HERE/'Dockerfile').read_bytes())
(HERE/'sources.json').write_text(json.dumps(sources,indent=2))
with (context.parent/'build.log').open('w') as log:
    subprocess.run(['docker','build','--progress=plain','-t','world/robocasa:1.0.1',str(context)],stdout=log,stderr=subprocess.STDOUT,check=True)
(context.parent/'image-inspect.json').write_bytes(subprocess.check_output(['docker','image','inspect','world/robocasa:1.0.1']))
