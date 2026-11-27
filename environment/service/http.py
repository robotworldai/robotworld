"""Authenticated episode transport shared by native simulator adapters."""
import hmac
import json
import os
import signal
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from environment.service.episode import Episode

def serve_episode(adapter, horizon, output, host='0.0.0.0', port=8765):
    output = Path(output)
    token = os.environ['WORLD_EPISODE_TOKEN']
    if len(token) < 24: raise ValueError('Use a random episode token of at least 24 characters')
    output.mkdir(parents=True, exist_ok=True)
    journal = (output/'events.jsonl').open('a', buffering=1)
    def record(kind, data):
        journal.write(json.dumps({'event':kind, 'data':data})+'\n')
    raw_limit = os.environ.get('WORLD_CONTROL_STEP_LIMIT', '').strip()
    control_step_limit = int(raw_limit) if raw_limit else None
    setting = os.environ.get('WORLD_CODE_CONTROL', '0')
    if setting not in ('0','1'): raise ValueError('WORLD_CODE_CONTROL must be 0 or 1')
    if setting == '1':
        from environment.service.feedback import FeedbackAdapter
        adapter = FeedbackAdapter(adapter, record)
    episode = Episode(adapter, horizon, str(uuid.uuid4()), record, control_step_limit=control_step_limit)
    # Simulator-owned evidence; never supplied by the agent's writable workspace.
    (output/'execution-config.json').write_text(json.dumps({
        'episode_id': episode.episode_id, 'native_horizon': horizon,
        'control_step_limit': control_step_limit, 'code_control': setting == '1',
        'observation_history': episode.describe().get('observation_history')}, indent=2))
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def reply(self, code, data):
            raw = json.dumps(data, allow_nan=False).encode()
            self.send_response(code); self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
        def authorized(self):
            return hmac.compare_digest(self.headers.get('Authorization',''), 'Bearer '+token)
        def do_GET(self):
            if not self.authorized(): return self.reply(401, {'error':'Unauthorized'})
            try:
                methods = {'/v1/describe':episode.describe, '/v1/observe':episode.observe, '/v1/status':episode.status}
                if self.path.startswith('/v1/receipts/'):
                    result = episode.receipt(self.path.rsplit('/',1)[-1])
                else: result = methods[self.path]()
                self.reply(200, result)
            except KeyError: self.reply(404, {'error':'Not found'})
        def do_POST(self):
            if not self.authorized(): return self.reply(401, {'error':'Unauthorized'})
            if self.path != '/v1/execute': return self.reply(404, {'error':'Not found'})
            try:
                length = int(self.headers.get('Content-Length',0))
                if not 0 < length <= 1048576: raise ValueError('Invalid body length')
                result = episode.execute(json.loads(self.rfile.read(length)))
                (output/'episode.json').write_text(json.dumps(episode.status(),indent=2))
                self.reply(200,result)
            except (ValueError, TypeError) as exc: self.reply(400, {'error':str(exc)})
    server = HTTPServer((host, port), Handler)
    print(json.dumps({'ready':True, **episode.status()}),flush=True)
    finalized = False
    def finalize(*_):
        nonlocal finalized
        if finalized: return
        with episode.lock:
            if episode.reason is None: episode.reason = 'agent_stopped'
            (output/'episode.json').write_text(json.dumps(episode.status(),indent=2))
            adapter.close()
            (output/'episode.json').write_text(json.dumps(episode.status(),indent=2))
            (output/'finalized').write_text('complete\n')
            finalized = True
    # Python dispatches signals on this thread, including inside native step().
    # Never close a simulator reentrantly while a request is mutating it.
    finalize_requested = False
    stop_requested = False
    def request_finalize(*_):
        nonlocal finalize_requested
        finalize_requested = True
    def request_stop(*_):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGUSR1, request_finalize)
    (output/'episode.json').write_text(json.dumps(episode.status(),indent=2))
    signal.signal(signal.SIGTERM, request_stop)
    server.timeout = 0.2
    try:
        while not stop_requested:
            server.handle_request()
            if finalize_requested:
                finalize()
    except KeyboardInterrupt: pass
    finally:
        finalize(); journal.close(); server.server_close()
        (output/'episode.json').write_text(json.dumps(episode.status(),indent=2))

