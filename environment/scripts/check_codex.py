"""Verify the source-built app-server and dynamic tool registration, without inference."""
import argparse
import json
from pathlib import Path

from environment.runtime.codex_session import CodexSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model')
    args = parser.parse_args()
    with CodexSession(args.manifest, args.output) as session:
        params = {'ephemeral': True, 'approvalPolicy': 'never', 'environments': [],
                  'baseInstructions': 'Robot tool interface validation.',
                  'dynamicTools': [{'type': 'function', 'name': 'move_eef',
                    'description': 'Registration check only; no robot is attached.',
                    'inputSchema': {'type': 'object', 'properties': {
                        'targets': {'type': 'object', 'additionalProperties': {'type': 'number'}},
                        'note': {'type': 'string'}}, 'required': ['targets', 'note']}}]}
        if args.model:
            params['model'] = args.model
        response = session.rpc('thread/start', params)
        result = {'status': 'registered', 'thread_id': response['thread']['id'],
                  'commit': session.manifest['commit'], 'inference_requested': False}
        (args.output / 'protocol-check.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))


if __name__ == '__main__':
    main()
