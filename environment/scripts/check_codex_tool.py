"""Real Codex inference/tool round trip, explicitly WITHOUT a simulator.

This isolates agent transport from GPU/Isaac failures. It is not a robot test.
"""
import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image

from environment.runtime.episode_runner import run_episode


class ProtocolProbe:
    instructions = (
        'This is a software tool protocol test. No robot or physical scene is attached. '
        'The image is a synthetic red square. Call move_eef exactly once with '
        'targets={"left_z":0.9}, note="protocol check", then finish. '
        'Do not call other tools and do not claim a real robot moved.')
    task_instruction = instructions

    def tool_spec(self):
        return {'type': 'function', 'name': 'move_eef',
                'description': 'Protocol validation only. This handler cannot actuate a robot.',
                'inputSchema': {'type': 'object', 'properties': {
                    'targets': {'type': 'object', 'additionalProperties': {'type': 'number'}},
                    'note': {'type': 'string'}}, 'required': ['targets', 'note']}}

    def observe_content(self):
        buffer = io.BytesIO()
        Image.new('RGB', (32, 32), (255, 0, 0)).save(buffer, format='PNG')
        return [{'type': 'inputText', 'text': 'Protocol probe: synthetic image, no simulator.'},
                {'type': 'inputImage', 'imageUrl': 'data:image/png;base64,' +
                 base64.b64encode(buffer.getvalue()).decode()}]

    def move_eef(self, arguments):
        if arguments.get('targets') != {'left_z': .9}:
            raise ValueError('Protocol probe received an unexpected target')
        return {'success': True, 'contentItems': self.observe_content()}, {
            'protocol_only': True, 'executed_waypoints': 0, 'robot_moved': False}

    def ended(self):
        return False

    def official_success(self):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model')
    parser.add_argument('--model-catalog', type=Path)
    args = parser.parse_args()
    result = run_episode(ProtocolProbe(), manifest=args.manifest, output_dir=args.output,
                         model=args.model, max_actions=1, timeout_s=120,
                         config_overrides=([f'model_catalog_json={json.dumps(str(args.model_catalog.resolve()))}']
                                           if args.model_catalog else []))
    print(json.dumps({'mode': 'protocol_only_no_simulator',
                      'action_calls': result['action_calls'], 'termination': result['termination']}))
    if result['action_calls'] != 1 or result['termination'] != 'agent_completed':
        raise SystemExit('Real Codex tool round trip did not complete')


if __name__ == '__main__':
    main()
