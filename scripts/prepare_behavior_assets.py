"""Prepare all selected BEHAVIOR official instances after the user's own agreement."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--accept-license',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--image',default='stanfordvl/behavior:3.9.3');a=p.parse_args()
    if not a.accept_license:p.error('Read official BEHAVIOR asset terms and personally accept with --accept-license; no credentials are redistributed')
    target=ROOT/'Assets/behavior_1k/data';legacy=ROOT/'var/datasets/behavior_1k'
    if not a.dry_run:
        target.mkdir(parents=True,exist_ok=True);legacy.parent.mkdir(parents=True,exist_ok=True)
        if legacy.exists() and legacy.resolve()!=target.resolve():raise RuntimeError('Existing legacy data; migrate it before this command')
        if not legacy.exists():legacy.symlink_to(os.path.relpath(target,legacy.parent),target_is_directory=True)
    rows=json.loads((ROOT/'environment/evaluation/suites.json').read_text())['behavior_1k']['cases']
    commands=[]
    for r in rows:
        commands.append([sys.executable,'-m','environment.datasets.behavior_1k.prepare_scene','--destination',str(target),'--cache',str(ROOT/'var/cache/behavior-zip-index'),'--task-name',r['task'],'--instance-id',str(r['instance_id']),'--accept-license'])
    commands.append(['docker','run','--rm','--mount',f'type=bind,src={target},dst=/data','-e','OMNIGIBSON_DATA_PATH=/data','-e','OMNIGIBSON_NO_OMNIVERSE=1',a.image,'python','-c',
        'from pathlib import Path; assert Path("/data/behavior-1k-assets/VERSION").is_file(); from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)'])
    for cmd in commands:
        print(json.dumps(cmd),flush=True)
        if not a.dry_run:subprocess.run(cmd,cwd=ROOT,check=True)
if __name__=='__main__':main()
