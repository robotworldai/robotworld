"""Copy an explicit Codex model catalog, changing only tool presentation mode."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve() or args.output.exists():
        raise ValueError('Use a new output file; never overwrite the original catalog')
    catalog = json.loads(args.source.read_text())
    matches = [m for m in catalog['models'] if m.get('slug') == args.model]
    if len(matches) != 1:
        raise ValueError('Selected model must exist exactly once in the source catalog')
    original = matches[0].get('tool_mode')
    matches[0]['tool_mode'] = 'direct'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, indent=2) + '\n')
    print(json.dumps({'model': args.model, 'previous_tool_mode': original,
                      'tool_mode': 'direct', 'output': str(args.output.resolve())}))


if __name__ == '__main__':
    main()
