"""Re-evaluate completed custom-driving traces with frozen scoring sources."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

WORLD = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    root = args.campaign / 'wheeledlab'
    for run in sorted(root.glob('rw-*')):
        if not (run / 'result.json').exists():
            rows.append({'run': str(run.resolve()), 'case': run.name, 'passed': False, 'reason': 'No completed result'})
            continue
        precision = run.name in ('rw-twin-beam', 'rw-reverse-bay', 'rw-parallel-park')
        file = 'precision_replay.py' if precision else 'replay_score.py'
        cmd = [sys.executable, str(WORLD / 'third_party/benchmarks/wheeledlab/robotworld' / file), str(run), '--check']
        process = subprocess.run(cmd, capture_output=True, text=True, cwd=WORLD)
        (run / 'score-replay.log').write_text(process.stdout + process.stderr)
        row = {'run': str(run.resolve()), 'case': run.name, 'passed': process.returncode == 0,
               'frozen_score_replay_returncode': process.returncode}
        if precision and process.returncode == 0:
            independent = subprocess.run([sys.executable, '-m', 'environment.validation.driving_line_audit', str(run)],
                                         capture_output=True, text=True, cwd=WORLD)
            (run / 'independent-line-audit.log').write_text(independent.stdout + independent.stderr)
            evidence = json.loads((run / 'independent-line-audit.json').read_text()) if (run / 'independent-line-audit.json').exists() else {}
            row['line_audit_matches'] = evidence.get('matches')
            row['passed'] = row['passed'] and independent.returncode == 0 and evidence.get('matches') is True
        rows.append(row)
    result = {'scope': 'Seven custom courses only. Recorded-state scoring consistency, independent paint-area / bridge tread-interval checks; not proof of all collision states or physical tire contact.',
              'expected_cases': 7, 'cases': rows,
              'passed': len(rows) == 7 and all(row['passed'] for row in rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
