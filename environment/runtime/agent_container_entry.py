"""Bridge container loopback to a host-owned, credential-free model socket."""
import json
from contextlib import suppress
import socket
import socketserver
import subprocess
import threading
import sys
from pathlib import Path


class Relay(socketserver.BaseRequestHandler):
    upstream_socket = '/model-relay/relay.sock'

    def handle(self):
        with socket.socket(socket.AF_UNIX) as target:
            target.connect(self.upstream_socket)
            self.request.settimeout(420)
            target.settimeout(420)
            # Read responses while uploading: an early HTTP 413 must not be lost
            # behind a blocked send of a large request body.
            def upload():
                try:
                    while data := self.request.recv(65536):
                        target.sendall(data)
                except OSError:
                    pass
                finally:
                    with suppress(OSError):
                        target.shutdown(socket.SHUT_WR)
            writer = threading.Thread(target=upload, daemon=True)
            writer.start()
            try:
                while data := target.recv(65536):
                    self.request.sendall(data)
            except OSError:
                pass
            finally:
                with suppress(OSError):
                    self.request.shutdown(socket.SHUT_RDWR)
                with suppress(OSError):
                    target.shutdown(socket.SHUT_RDWR)
                writer.join(timeout=2)


def main():
    with socketserver.ThreadingTCPServer(('127.0.0.1', 0), Relay) as server:
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        provider = dict(name='Host-owned model relay',
                        base_url=f'http://127.0.0.1:{server.server_address[1]}/v1',
                        wire_api='responses', requires_openai_auth=False,
                        supports_websockets=False, stream_idle_timeout_ms=360000)
        # TOML inline table, not JSON object syntax.
        value = '{' + ','.join(k+'='+json.dumps(v) for k, v in provider.items()) + '}'
        command = ['/runtime/codex-app-server', '--listen', 'stdio://',
                   '-c', 'model_provider="world_relay"',
                   '-c', 'model_providers.world_relay='+value,
                   '-c', 'check_for_update_on_startup=false']
        if Path('/runtime/model.json').exists():
            config = json.loads(Path('/runtime/model.json').read_text())
            for key in ('model', 'model_reasoning_effort'):
                if key in config:
                    command += ['-c', key+'='+json.dumps(config[key])]
        if Path('/runtime/models.json').exists():
            command += ['-c', 'model_catalog_json="/runtime/models.json"']
        return subprocess.call(command+sys.argv[1:], cwd='/workspace')


if __name__ == '__main__':
    raise SystemExit(main())
