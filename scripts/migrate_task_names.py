#!/usr/bin/env python3
"""Migrate result metadata to descriptive names; preserve raw traces and old path aliases."""
import argparse
import json
from pathlib import Path
import sys

WORLD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORLD))
from environment.evaluation.task_names import entries, named_record
from environment.evaluation.rollout_results import save_json, write_summary
from environment.evaluation.resume import batch_lock


def migrate(root, apply=False):
    changes = []
    for task in entries():
        if task['existing_integration']: continue
        bench, old, new = task['project'], task['id'], task['name']
        folder = root/bench; source = folder/old; dest = folder/new
        if not source.exists(): continue
        if source.is_symlink():
            if source.resolve() != dest.resolve(): raise ValueError('Unexpected task alias: '+str(source))
        else:
            for path in source.glob('run-*/run.json'):
                if json.loads(path.read_text()).get('status') == 'running':
                    raise ValueError('Do not rename a running task: '+str(path))
            if dest.exists(): raise ValueError('Destination already exists; no automatic merge: '+str(dest))
            changes.append({'from':str(source),'to':str(dest)})
            if apply:
                source.rename(dest)
                source.symlink_to(dest.name, target_is_directory=True)
        if not apply: continue
        for path in dest.glob('run-*/run.json'):
            record = named_record(json.loads(path.read_text()))
            record['path'] = str(path.parent.resolve())
            save_json(path,record)
        # Completed benchmark summaries are rewritten from authoritative run.json.
        # A running batch is locked; refuse rather than race its statistics writer.
        for path in (folder/'summaries').glob('*.json'):
            with batch_lock(path.with_suffix('.lock')):
                summary = json.loads(path.read_text())
                records = []
                for row in summary['runs']:
                    rp = Path(row['path'])/'run.json'
                    record = named_record(json.loads(rp.read_text()) if rp.exists() else row)
                    if Path(record['path']).exists(): record['path'] = str(Path(record['path']).resolve())
                    records.append(record)
                write_summary(folder,summary['batch'],records,summary['configuration'])
    return changes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,default=WORLD/'outputs')
    p.add_argument('--apply',action='store_true')
    args=p.parse_args()
    changes=migrate(args.output_root.resolve(),args.apply)
    print(json.dumps(changes,ensure_ascii=False,indent=2))
    if args.apply:
        save_json(args.output_root/'_task-name-migration.json',{'changes':changes,'old_directories':'relative symlinks; raw simulator/Codex artifacts untouched'})

if __name__=='__main__':main()
