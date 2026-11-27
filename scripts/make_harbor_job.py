"""Build a standard Harbor job with independently selected tasks, agent and model."""
import argparse
from pathlib import Path
import sys
import tomllib


def build_job(root, *, tasks, agent, agent_import_path, model, attempts=1, concurrency=1):
    if bool(agent) == bool(agent_import_path):
        raise ValueError('Choose exactly one built-in agent or custom agent import path')
    if not model or attempts < 1 or concurrency < 1:
        raise ValueError('Model, positive attempts and positive concurrency are required')
    root = Path(root).resolve()
    selected = sorted((root/'tasks').glob('*/task.toml')) if tasks == ['all'] else [root/'tasks'/name/'task.toml' for name in tasks]
    if not selected: raise ValueError('No tasks selected')
    entries = []
    for path in selected:
        path = path.resolve()
        if not path.is_relative_to(root/'tasks') or not path.is_file():
            raise ValueError('Unknown task: '+str(path))
        with path.open('rb') as stream: tomllib.load(stream)
        entry = {'path':str(path.parent)}
        if entry not in entries: entries.append(entry)
    selection = {'name':agent} if agent else {'import_path':agent_import_path}
    selection['model_name'] = model
    return {'jobs_dir':str(root/'outputs/harbor'), 'n_attempts':attempts,
            'n_concurrent_trials':concurrency, 'agents':[selection], 'tasks':entries}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--agent',help='Built-in Harbor agent name')
    group.add_argument('--agent-import-path',help='Custom Harbor Agent class, module:Class')
    p.add_argument('--model',required=True,help='Harbor provider/model identifier; compatibility depends on the harness')
    p.add_argument('--task',action='append',required=True,help='Task directory name, repeatable; all selects the full suite')
    p.add_argument('--attempts',type=int,default=1)
    p.add_argument('--concurrency',type=int,default=1)
    p.add_argument('--agent-version',help='Optional built-in agent version pin')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    job=build_job(Path(__file__).resolve().parents[1],tasks=args.task,agent=args.agent,
                  agent_import_path=args.agent_import_path,model=args.model,
                  attempts=args.attempts,concurrency=args.concurrency)
    if args.agent_version: job['agents'][0]['kwargs']={'version':args.agent_version}
    import yaml
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(yaml.safe_dump(job,sort_keys=False))
    print(str(args.output))

if __name__=='__main__': main()
