"""Loopback, stateless Responses relay; no agent, account login or tool execution."""
import ipaddress
from urllib.parse import urlsplit
import hmac
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from .observation_window import trim_observations

def permitted_model_endpoint(url):
    parsed = urlsplit(url)
    if parsed.scheme == 'https': return True
    if parsed.scheme != 'http': return False
    try: address = ipaddress.ip_address(parsed.hostname)
    except ValueError: return False
    if address.is_loopback: return True
    return (os.environ.get('WORLD_ALLOW_PRIVATE_MODEL_HTTP') == '1'
            and any(address in net for net in (ipaddress.ip_network('10.0.0.0/8'),
                ipaddress.ip_network('172.16.0.0/12'), ipaddress.ip_network('192.168.0.0/16'))))

class ObservationWindowProxy:
    def __init__(self,upstream,api_key_env,output,timeout=180):
        self.upstream=upstream.rstrip('/');self.key=os.environ[api_key_env]
        self.output=Path(output);self.timeout=timeout;self.lock=threading.Lock()
        if not permitted_model_endpoint(self.upstream): raise ValueError('HTTPS required; private HTTP requires explicit WORLD_ALLOW_PRIVATE_MODEL_HTTP=1')
    def __enter__(self):
        owner=self;self.output.parent.mkdir(parents=True,exist_ok=True)
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def error(self,status,message):
                raw=json.dumps({'error':{'message':message,'type':'image_window_error'}}).encode()
                self.send_response(status);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(raw)
            def do_POST(self):
                if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+owner.key):return self.error(401,'Unauthorized')
                if self.path not in ('/v1/responses','/v1/responses/compact'):return self.error(404,'Unsupported path')
                started=False
                try:
                    n=int(self.headers.get('Content-Length',0))
                    if not 0<n<=100*1024*1024:raise ValueError('Invalid request length')
                    body,stats=trim_observations(json.loads(self.rfile.read(n)))
                    stats.update(timestamp=time.time(),path=self.path,model=body.get('model'))
                    with owner.lock:
                        with owner.output.open('a') as f:f.write(json.dumps(stats)+'\n')
                    headers={'Content-Type':'application/json','Authorization':'Bearer '+owner.key}
                    for name in ('OpenAI-Beta','OpenAI-Organization','OpenAI-Project'):
                        if self.headers.get(name):headers[name]=self.headers[name]
                    req=urllib.request.Request(owner.upstream+self.path.removeprefix('/v1'),data=json.dumps(body).encode(),headers=headers)
                    try:response=urllib.request.urlopen(req,timeout=owner.timeout)
                    except urllib.error.HTTPError as e:response=e
                    with response:
                        self.send_response(response.status)
                        self.send_header('Content-Type',response.headers.get('Content-Type','application/json'))
                        self.end_headers();started=True
                        while chunk:=response.read1(65536):self.wfile.write(chunk);self.wfile.flush()
                except (ValueError,TypeError) as e:self.error(400,str(e))
                except (BrokenPipeError,ConnectionResetError):pass
                except Exception as e:
                    # Do not include credentials or request bodies in diagnostics.
                    try:
                        if not started:self.error(502,'Image-window upstream failure: '+type(e).__name__)
                    except (BrokenPipeError,ConnectionResetError):pass
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.base_url=f'http://127.0.0.1:{self.server.server_port}/v1';return self
    def __exit__(self,*args):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=5)
