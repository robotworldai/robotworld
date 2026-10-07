"""Replay a precision score from frozen private raw states, not model claims."""
import argparse,hashlib,importlib,json,sys,types
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
    folder=a.run/'protocol-source';hashes=json.loads((a.run/'protocol-hashes.json').read_text())
    for name in ['precision_scoring.py','scoring.py']:
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=hashes[name]:raise RuntimeError('Frozen source hash mismatch:'+name)
    pkg=types.ModuleType('_frozen_precision');pkg.__path__=[str(folder.resolve())];sys.modules[pkg.__name__]=pkg
    cls=importlib.import_module('_frozen_precision.precision_scoring').Evaluator
    judge=cls(json.loads((a.run/'scenario.json').read_text()))
    for line in (a.run/'events/scoring.jsonl').read_text().splitlines():
        row=json.loads(line);state=row['payload']['state']
        if row['kind']=='initial_state':judge.previous=state
        else:judge.update(state)
    print(json.dumps(judge.report(),indent=2))
    if a.check:
        assert judge.report()==json.loads((a.run/'result.json').read_text())['robotworld']
        print('MATCH: independently recomputed from frozen raw trajectory')


if __name__=='__main__':main()
