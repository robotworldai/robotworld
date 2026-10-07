"""Build selected benchmark images in dependency order, or pull supplied registry images."""
import argparse,json,os,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from environment.evaluation.rollout_catalog import catalog
from environment.runtime.native_project_launch import load_project
BASE='world/robodojo:isaac6.0.1-local'
def recipes():
    result={};selected={}
    suites=json.loads((ROOT/'environment/evaluation/suites.json').read_text())
    def add(image,file,context,command=None):
        result[image]={'file':ROOT/file,'command':command or ['docker','build','-t',image,'-f',file,context]}
        return image
    selected['robodojo']=BASE
    add(BASE,'environment/containers/robodojo/Dockerfile.isaac601.snapshot','.',[sys.executable,'environment/containers/robodojo/build_isaac601_snapshot.py'])
    add('world/robolab:0.3.1-isaac5.0','third_party/benchmarks/robolab/docker/Dockerfile','third_party/benchmarks/robolab/docker')
    selected['robolab']=add('world/robolab:0.3.1-isaac6.0.1-experimental','third_party/benchmarks/robolab/docker/Dockerfile.isaac601','third_party/benchmarks/robolab/docker')
    selected['behavior_1k']=add('world/behavior-1k:isaac6.0.1-experimental','environment/containers/behavior_1k/Dockerfile.isaac601','environment/containers/behavior_1k')
    # The suite registry is authoritative for core image tags.
    from environment.evaluation.runner import image
    from types import SimpleNamespace
    actual=image('behavior_1k',SimpleNamespace(runtime='isaac601'))
    result[actual]=result.pop(selected['behavior_1k']);selected['behavior_1k']=actual
    result[actual]['command']=['docker','build','--build-context','behavior_source=third_party/benchmarks/behavior_1k/checkout','-t',actual,'-f','environment/containers/behavior_1k/Dockerfile.isaac601','environment/containers/behavior_1k']
    selected['robocasa']=add('world/robocasa:1.0.1','third_party/benchmarks/robocasa/docker/Dockerfile','.',[sys.executable,'third_party/benchmarks/robocasa/docker/build.py'])
    for bench in ('humanoid_soccer','ai_cps','wheeledlab'):
        tag=suites[bench]['image'];folder=f'third_party/benchmarks/{bench}/docker'
        selected[bench]=add(tag,folder+'/Dockerfile',folder)
    for bench,tasks in catalog().items():
        if bench in selected:continue
        tags=[]
        for profile in dict.fromkeys(task['runtime_profile'] for task in tasks):
            folder,cfg=load_project(bench,profile)
            file=str((folder/cfg.get('build_dockerfile','docker/Dockerfile')).relative_to(ROOT))
            context=str((folder/cfg.get('build_context','docker')).relative_to(ROOT))
            tag=add(cfg['image'],file,context)
            if tag not in tags:tags.append(tag)
        selected[bench]=tags
    selected={bench:([tags] if isinstance(tags,str) else tags) for bench,tags in selected.items()}
    return result,selected

def local_dependencies(dockerfile):
    """Resolve global ARG defaults used by FROM in the bundled recipes."""
    defaults={};dependencies=[];before_from=True
    for line in dockerfile.splitlines():
        arg=re.match(r'^\s*ARG\s+(\w+)=(\S+)\s*$',line,re.I)
        if arg and before_from:defaults[arg[1]]=arg[2].strip('\"\'')
        base=re.match(r'^\s*FROM\s+(\S+)',line,re.I)
        if not base:continue
        before_from=False
        image=re.sub(r'\$\{(\w+)\}|\$(\w+)',lambda m:defaults.get(m[1] or m[2],m[0]),base[1])
        if '$' in image:raise ValueError('Unresolved Docker base image: '+image)
        if image.startswith('world/') and image not in dependencies:dependencies.append(image)
    return dependencies

def main():
    table,selected=recipes()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bench',action='append',choices=list(selected));p.add_argument('--dry-run',action='store_true');p.add_argument('--registry-map',type=Path,default=ROOT/'environment/containers/image-sources.json');a=p.parse_args()
    mapping=json.loads(a.registry_map.read_text()).get('images',{}) if a.registry_map.exists() else {}
    done=set()
    def ensure(tag):
        if tag in done:return
        done.add(tag)
        if not a.dry_run and subprocess.run(['docker','image','inspect',tag],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
            print('Present: '+tag,flush=True);return
        remote=mapping.get(tag)
        if remote:
            cmds=[['docker','pull',remote],['docker','tag',remote,tag]]
        else:
            if tag not in table:raise RuntimeError('No recipe for local dependency '+tag)
            item=table[tag]
            if tag==BASE and not a.dry_run:
                raise RuntimeError('Missing Isaac 6.0.1 base image. Set its authorized registry URL in environment/containers/image-sources.json, or build the parameterized local runtime snapshot described in docs/HANDOFF.md. No public image is assumed.')
            for dep in local_dependencies(item['file'].read_text()):ensure(dep)
            cmds=[item['command']]
        for cmd in cmds:
            print(json.dumps(cmd),flush=True)
            if not a.dry_run:subprocess.run(cmd,cwd=ROOT,check=True)
    for bench in a.bench or selected:
        for tag in selected[bench]:ensure(tag)
if __name__=='__main__':main()
