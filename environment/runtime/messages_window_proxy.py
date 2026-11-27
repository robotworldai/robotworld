"""External Messages image window; leaves the original Claude Code archive intact."""
from .observation_window_proxy import permitted_model_endpoint
import copy
import hmac
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

NOTICE = '[Earlier RobotWorld image omitted from this request; use the newest observation packet. Original image remains in the trajectory archive.]'


def image_count(value):
    if isinstance(value, list):
        return sum(map(image_count, value))
    if isinstance(value, dict):
        if value.get('type') == 'image':
            return 1
        return sum(image_count(x) for x in value.values())
    return 0


def trim_messages(body, max_images=None):
    """Recognize only paired RobotWorld MCP observation tool results."""
    out = copy.deepcopy(body)
    calls, packets = {}, []
    for message in out.get('messages', []):
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if message.get('role') == 'assistant' and part.get('type') == 'tool_use':
                identity = part.get('id')
                if identity in calls:
                    raise ValueError('Duplicate tool-use identifier')
                calls[identity] = part.get('name')
            if message.get('role') != 'user' or part.get('type') != 'tool_result':
                continue
            contents = part.get('content')
            if not isinstance(contents, list) or not image_count(contents):
                continue
            # Other tools, user images, and unpaired compacted history are untouched.
            if calls.get(part.get('tool_use_id')) != 'mcp__robotworld__observe':
                continue
            metadata = []
            for item in contents:
                if item.get('type') != 'text':
                    continue
                try:
                    data = json.loads(item.get('text', ''))
                except (ValueError, TypeError):
                    continue
                if (isinstance(data, dict) and data.get('protocol') == 'robotworld.episode.v1'
                        and isinstance(data.get('episode_id'), str) and data['episode_id']
                        and type(data.get('control_step')) is int and data['control_step'] >= 0
                        and isinstance(data.get('observation'), dict)
                        and data.get('observation_id') == f"{data['episode_id']}:{data['control_step']}"):
                    metadata.append(data)
            if len(metadata) != 1:
                raise ValueError('RobotWorld image tool result lacks unique episode observation metadata')
            packets.append((part, metadata[0]))
    keep = None
    if packets:
        if len({meta['episode_id'] for _, meta in packets}) != 1:
            raise ValueError('Mixed RobotWorld episodes in Messages history')
        keep = max(range(len(packets)), key=lambda i: (packets[i][1]['control_step'], i))
        for i, (part, _) in enumerate(packets):
            if i != keep:
                part['content'] = [dict(type='text', text=NOTICE) if p.get('type') == 'image' else p
                                   for p in part['content']]
    after = image_count(out)
    if max_images is not None and after > max_images:
        raise ValueError('Images exceed explicitly configured cap; other images were not removed')
    return out, dict(images_before=image_count(body), images_after=after, packets_before=len(packets),
                     kept_control_step=packets[keep][1]['control_step'] if keep is not None else None,
                     kept_tool_use_id=packets[keep][0]['tool_use_id'] if keep is not None else None)


class MessagesWindowProxy:
    def __init__(self, upstream, api_key, output, timeout=180, max_images=None):
        self.upstream = upstream.rstrip('/')
        if self.upstream.endswith('/v1'):
            self.upstream = self.upstream[:-3]
        if not permitted_model_endpoint(self.upstream):
            raise ValueError('HTTPS upstream or container-loopback converter required')
        self.key, self.output, self.timeout = api_key, Path(output), timeout
        self.max_images = max_images
        self.lock = threading.Lock()

    def __enter__(self):
        owner = self
        self.output.parent.mkdir(parents=True, exist_ok=True)
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def error(self, status, message):
                raw = json.dumps({'type':'error', 'error':{'type':'image_window_error', 'message':message}}).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(raw)
            def do_POST(self):
                if not (hmac.compare_digest(self.headers.get('x-api-key', ''), owner.key) or
                        hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer '+owner.key)):
                    return self.error(401, 'Unauthorized')
                if urllib.parse.urlsplit(self.path).path not in ('/v1/messages', '/v1/messages/count_tokens'):
                    return self.error(404, 'Unsupported Messages path')
                started = False
                try:
                    length = int(self.headers.get('Content-Length', 0))
                    if not 0 < length <= 100 * 1024 * 1024:
                        raise ValueError('Invalid body length')
                    body, stats = trim_messages(json.loads(self.rfile.read(length)), owner.max_images)
                    stats.update(timestamp=time.time(), path=self.path, model=body.get('model'))
                    with owner.lock:
                        with owner.output.open('a') as log:
                            log.write(json.dumps(stats)+'\n')
                    headers = {'Content-Type':'application/json', 'Accept-Encoding':'identity'}
                    for name in self.headers:
                        if name.lower() in ('x-api-key', 'authorization', 'accept') or name.lower().startswith('anthropic-'):
                            headers[name] = self.headers[name]
                    request = urllib.request.Request(owner.upstream+self.path, data=json.dumps(body).encode(), headers=headers)
                    try:
                        response = urllib.request.urlopen(request, timeout=owner.timeout)
                    except urllib.error.HTTPError as exc:
                        response = exc
                    with response:
                        self.send_response(response.status)
                        self.send_header('Content-Type', response.headers.get('Content-Type', 'application/json'))
                        if response.headers.get('request-id'):
                            self.send_header('request-id', response.headers['request-id'])
                        self.end_headers()
                        started = True
                        while chunk := response.read1(65536):
                            self.wfile.write(chunk)
                            self.wfile.flush()
                except (ValueError, TypeError, KeyError) as exc:
                    self.error(400, str(exc))
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as exc:
                    if not started:
                        self.error(502, 'Messages window upstream failure: '+type(exc).__name__)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f'http://127.0.0.1:{self.server.server_port}'
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
