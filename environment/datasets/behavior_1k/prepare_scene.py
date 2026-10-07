"""Prepare one official task instance and its R1Pro dependencies.

Requires user acceptance of the upstream asset license. The encryption key is
installed separately by the unmodified official tool inside the official image.
"""
import argparse
import csv
import json
import zipfile
from pathlib import Path
from .remote_zip import RangeFile,resolve_archive
from .subset import fetch
from .systems import select_system_files,prepare_registry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--task-name',default='carrying_in_groceries')
    p.add_argument('--instance-id',type=int,default=311)
    p.add_argument('--accept-license',action='store_true',help='Only after personally accepting the official BEHAVIOR asset terms')
    a=p.parse_args()
    if not a.accept_license:p.error('Read and accept the upstream asset license before preparing assets')
    if a.instance_id<0:p.error('Instance ID must be nonnegative')
    a.destination.mkdir(parents=True,exist_ok=True);a.cache.mkdir(parents=True,exist_ok=True)
    catalog=Path(__file__).resolve().parents[3]/'third_party/benchmarks/behavior_1k/checkout/docs/challenge/task_data.json'
    task=next((t for t in json.loads(catalog.read_text())['tasks'] if t['id']==a.task_name),None)
    if task is None:p.error('Task must exist in the pinned official task catalog')
    reports=a.destination/'subsets'/a.task_name/str(a.instance_id)
    reports.mkdir(parents=True,exist_ok=True)
    archives=['2026-challenge-task-instances.zip','omnigibson-robot-assets-3.8.2.zip','behavior-1k-assets-3.9.0.zip']
    for name in archives:
        ip=a.cache/(name+'.json')
        if not ip.exists():
            url,rev=resolve_archive(name);raw=RangeFile(url)
            with zipfile.ZipFile(raw) as z:
                records=[{'path':i.filename,'size':i.file_size,'compressed':i.compress_size} for i in z.infolist()]
            ip.write_text(json.dumps({'url':url,'revision':rev,'archive_bytes':raw.size,'index_transfer_bytes':raw.transferred,'files':records}))
    idx=a.cache/(archives[0]+'.json');files=json.loads(idx.read_text())['files']
    selected=[f['path'] for f in files if f['path'].startswith('metadata/') or
              (f'_task_{a.task_name}_' in f['path'] and ('_0_0_template' in f['path'] or f'_0_{a.instance_id}_template' in f['path']))]
    if not any(f'_task_{a.task_name}_0_{a.instance_id}_template-tro_state.json' in name for name in selected):
        raise ValueError('Requested task instance is absent from the official archive')
    report=fetch(idx,a.destination/'2026-challenge-task-instances',selected)
    (reports/'task-subset-manifest.json').write_text(json.dumps(report,indent=2))
    # Official current room metadata, not the older *_partial_rooms snapshot:
    # that snapshot omits garden/corridor/living-room assets now included by evaluator.
    task_root=a.destination/'2026-challenge-task-instances'
    with (task_root/'metadata/B100_task_misc.csv').open() as f:
        row=next(r for r in csv.DictReader(f) if r['Task']==a.task_name)
    rooms=set(row['Rooms to inlcude'].splitlines())
    # InteractiveTraversableScene._should_load_object in pinned v3.9.3.
    structures={'walls','ceilings','fence','roof','background','door','sliding_door'}
    models=set()
    for path in task_root.rglob(f'*_task_{a.task_name}_*_template.json'):
        if not any(path.name.endswith(f'_0_{i}_template.json') for i in (0,a.instance_id)):continue
        for obj in json.loads(path.read_text())['objects_info']['init_info'].values():
            args=obj['args']; assigned=args.get('in_rooms',[])
            if isinstance(assigned,str): assigned=[assigned]
            if args.get('category') not in structures and not rooms.intersection(assigned):continue
            if 'category' in args and 'model' in args:models.add(f"objects/{args['category']}/{args['model']}/")
    (reports/'scene-dependencies.json').write_text(json.dumps({'task':a.task_name,'instance_id':a.instance_id,'rooms':sorted(rooms),'models':sorted(models)},indent=2))
    if not models:raise RuntimeError('Task snapshot has no dependency models')
    for name,dest,prefixes in [
        (archives[1],'omnigibson-robot-assets',['models/r1pro/','models/background/','fonts/']),
        (archives[2],'behavior-1k-assets',sorted(models)+['systems/water/','metadata/',f"scenes/{task['scene_model']}/layout/",f"scenes/{task['scene_model']}/json/"])]:
        idx=a.cache/(name+'.json');files=json.loads(idx.read_text())['files']
        selected=[f['path'] for f in files if f['path']=='VERSION' or any(f['path'].startswith(p) for p in prefixes)]
        report=fetch(idx,a.destination/dest,selected)
        (reports/(dest+'-manifest.json')).write_text(json.dumps(report,indent=2))
    idx=a.cache/(archives[2]+'.json')
    taxonomy=Path(__file__).resolve().parents[3]/'third_party/benchmarks/behavior_1k/checkout/bddl3/bddl/generated_data/output_hierarchy_properties.json'
    systems,selected=select_system_files({p.split('/')[1] for p in models},taxonomy,json.loads(idx.read_text())['files'])
    report=fetch(idx,a.destination/'behavior-1k-assets',selected)
    (reports/'systems-manifest.json').write_text(json.dumps(report,indent=2))
    (reports/'system-dependencies.json').write_text(json.dumps(systems,indent=2))
    prepare_registry(idx,a.destination)
    print('Selected task/robot assets prepared. Install official key, then validate simulation dependency closure.')


if __name__=='__main__':main()
