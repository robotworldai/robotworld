"""Recompute success from private logged states using that run's frozen scorer."""
import argparse,importlib.util,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
    src=a.run/'protocol-source/scoring.py'
    if not src.exists():raise RuntimeError('No frozen scorer: cannot claim an exact replay')
    import hashlib
    hashes=json.loads((a.run/'protocol-hashes.json').read_text())
    if hashlib.sha256(src.read_bytes()).hexdigest()!=hashes['scoring.py']:raise RuntimeError('Frozen scoring code checksum mismatch')
    spec=importlib.util.spec_from_file_location('frozen_robotworld_scorer',src);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    evaluator=module.Evaluator(json.loads((a.run/'scenario.json').read_text()))
    for line in (a.run/'events/scoring.jsonl').read_text().splitlines():
        event=json.loads(line)
        if event['kind']=='initial_state':evaluator.previous=event['payload']['state']
        elif event['kind']=='scoring_step':evaluator.update(event['payload']['state'])
    result=evaluator.report();print(json.dumps(result,indent=2,ensure_ascii=False))
    if a.check:
        saved=json.loads((a.run/'result.json').read_text())['robotworld']
        if result!=saved:raise RuntimeError('Replayed score differs from saved result')
        print('MATCH: independently recomputed from recorded states')

if __name__=='__main__':main()
