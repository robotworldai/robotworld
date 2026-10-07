#!/usr/bin/env python3
"""Check streaming, image input, and tool round trips without starting a simulator."""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import sys
import tomllib
import urllib.request
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environment.runtime.model_config import filtered_config
from environment.runtime.request_audit import NoRedirect


def request(url, headers, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    opener = urllib.request.build_opener(NoRedirect)
    with opener.open(req, timeout=90) as response:
        if 'text/event-stream' not in response.headers.get('Content-Type', ''):
            raise ValueError('The provider did not return a streaming Responses event stream')
        for raw in response:
            if not raw.startswith(b'data:'):
                continue
            data = raw[5:].strip()
            if data == b'[DONE]':
                continue
            event = json.loads(data)
            if event.get('type') in {'error', 'response.failed', 'response.incomplete'}:
                raise ValueError('The provider returned a failed or incomplete response')
            if event.get('type') == 'response.completed':
                result = event['response']
                if result.get('status') not in (None, 'completed'):
                    raise ValueError('Response was not completed')
                return result
    raise ValueError('Stream ended without response.completed')


def check(home):
    config = filtered_config(tomllib.loads((Path(home) / 'config.toml').read_text()))
    provider = config['model_providers'][config['model_provider']]
    url = provider['base_url'].rstrip('/') + '/responses'
    headers = {'Content-Type': 'application/json', 'Accept': 'text/event-stream', **provider.get('http_headers', {})}
    token = provider.get('experimental_bearer_token')
    if token:
        headers['Authorization'] = 'Bearer ' + token
    buf = io.BytesIO()
    Image.new('RGB', (32, 32), (0, 255, 0)).save(buf, format='PNG')
    image = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    inputs = [{'role': 'user', 'content': [
        {'type': 'input_text', 'text': 'Inspect the image. Call record_colour with its dominant colour. After receiving the tool result, reply with OK.'},
        {'type': 'input_image', 'image_url': image}]}]
    tool = {'type': 'function', 'name': 'record_colour', 'description': 'Record the dominant colour seen in the image.',
            'parameters': {'type': 'object', 'properties': {'colour': {'type': 'string', 'enum': ['red', 'green', 'blue']}},
                           'required': ['colour'], 'additionalProperties': False}, 'strict': True}
    payload = {'model': config['model'], 'input': inputs, 'tools': [tool], 'stream': True, 'store': False,
               'tool_choice': {'type': 'function', 'name': 'record_colour'}}
    if config.get('model_reasoning_effort'):
        payload['reasoning'] = {'effort': config['model_reasoning_effort']}
    first = request(url, headers, payload)
    calls = [x for x in first.get('output', []) if x.get('type') == 'function_call']
    if len(calls) != 1 or calls[0].get('name') != 'record_colour' or json.loads(calls[0]['arguments']).get('colour') != 'green':
        raise ValueError('Image/tool check failed: expected one record_colour call identifying green')
    payload['input'] = inputs + first['output'] + [{'type': 'function_call_output', 'call_id': calls[0]['call_id'], 'output': 'Recorded successfully. Reply OK.'}]
    payload['tool_choice'] = 'none'
    second = request(url, headers, payload)
    answer = ''.join(c.get('text', '') for x in second.get('output', []) if x.get('type') == 'message' for c in x.get('content', []))
    if 'OK' not in answer.upper():
        raise ValueError('Tool-result round trip did not produce the expected acknowledgement')
    return {'streaming': True, 'image_input': True, 'tool_round_trip': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config-dir', type=Path, default=Path(os.environ.get('CODEX_AUTH_HOME', 'var/auth/api')))
    a = p.parse_args()
    try:
        print(json.dumps(check(a.config_dir)))
    except Exception as e:
        # Provider error bodies and URLs may contain credentials; never echo them.
        print('API check failed (' + type(e).__name__ + '). Check access, model ID, and streaming Responses/image/tool support.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
