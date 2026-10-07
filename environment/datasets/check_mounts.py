"""Read relocated asset samples through actual Docker bind mounts; no GPU/model."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

WORLD = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='world/robocasa:1.0.1')
    parser.add_argument('--output', type=Path, default=WORLD/'reports/assets/container-mounts.json')
    args = parser.parse_args()
    rows = []
    command = ['docker', 'run', '--rm', '--network', 'none', '--read-only', '--entrypoint', 'python']
    layout = json.loads((WORLD/'environment/datasets/asset-layout.json').read_text())
    for index, item in enumerate(layout['external']):
        source, target = WORLD/item['source'], WORLD/item['target']
        if not source.is_symlink() or source.resolve() != target.resolve() or not target.is_dir():
            raise RuntimeError('Expected completed canonical migration: '+str(source))
        files = [p for p in target.rglob('*') if p.is_file() and not p.is_symlink()]
        if not files:
            raise RuntimeError('Empty asset directory: '+str(target))
        # Prefer substantial resources; never print file contents or credentials.
        sample = max(files, key=lambda p:p.stat().st_size)
        digest = hashlib.sha256()
        with sample.open('rb') as stream:
            while block := stream.read(4*1024*1024):digest.update(block)
        destination = '/asset-mounts/'+str(index)
        command += ['--mount', f'type=bind,src={source},dst={destination},readonly']
        rows.append({'legacy_source':item['source'], 'canonical_target':item['target'],
                     'sample':str(sample.relative_to(target)), 'bytes':sample.stat().st_size,
                     'sha256':digest.hexdigest(), 'container_path':destination+'/'+str(sample.relative_to(target))})
    program = '''import hashlib,json,sys
rows=json.loads(sys.argv[1])
for row in rows:
 h=hashlib.sha256();size=0
 with open(row['container_path'],'rb') as stream:
  while True:
   block=stream.read(4*1024*1024)
   if not block:break
   h.update(block);size+=len(block)
 row['container_sha256']=h.hexdigest();row['passed']=size==row['bytes'] and h.hexdigest()==row['sha256']
print(json.dumps(rows))
'''
    command += [args.image, '-c', program, json.dumps(rows)]
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    checked = json.loads(process.stdout)
    passed = len(checked)==len(rows) and all(row['passed'] for row in checked)
    result = {'passed':passed, 'image':args.image, 'gpu':False, 'model_calls':0,
              'scope':'One largest regular-file sample per relocated directory through Docker bind mount; '
                      'full directory hashes are separately checked during migration.', 'mounts':checked}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'passed':passed, 'mounts':len(checked)}))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
