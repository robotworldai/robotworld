"""Public API setup and protocol checks use only a loopback test server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import tomllib
import pytest
from scripts.configure_api import configure
from scripts.check_api import check


def test_configuration_never_serializes_key(tmp_path, monkeypatch):
    monkeypatch.setenv('WORLD_MODEL_API_KEY', 'private-test-value')
    path = configure(tmp_path / 'provider', 'https://example.invalid/v1/', 'custom-model')
    assert path.stat().st_mode & 0o077 == 0
    assert 'private-test-value' not in path.read_text()
    config = tomllib.loads(path.read_text())
    assert config['model'] == 'custom-model'
    assert config['model_providers']['api']['base_url'] == 'https://example.invalid/v1'
    with pytest.raises(FileExistsError):
        configure(path.parent, 'https://other.invalid/v1', 'other-model')
    assert tomllib.loads(path.read_text()) == config


@pytest.mark.parametrize('url', ['https://user:secret@example.invalid/v1', 'https://example.invalid/v1?key=secret', 'file:///tmp/model'])
def test_reject_url_credentials(tmp_path, url):
    with pytest.raises(ValueError):
        configure(tmp_path, url, 'model')


def test_streaming_image_and_tool_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv('WORLD_MODEL_API_KEY', 'private-test-value')
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append((self.path, self.headers['Authorization'], body))
            if len(calls) == 1:
                output = [{'type': 'function_call', 'id': 'fc_test', 'call_id': 'call_test', 'name': 'record_colour', 'arguments': '{"colour":"green"}'}]
            else:
                output = [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'OK'}]}]
            event = {'type': 'response.completed', 'response': {'status': 'completed', 'output': output}}
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            self.wfile.write(('data: ' + json.dumps(event) + '\n\n').encode())
    with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            configure(tmp_path, f'http://127.0.0.1:{server.server_port}/v1', 'custom-model')
            assert check(tmp_path) == {'streaming': True, 'image_input': True, 'tool_round_trip': True}
        finally:
            server.shutdown()
            thread.join()
    assert len(calls) == 2
    assert all(path == '/v1/responses' and auth == 'Bearer private-test-value' and body['model'] == 'custom-model' for path, auth, body in calls)
    assert calls[0][2]['input'][0]['content'][1]['image_url'].startswith('data:image/png;base64,')
    assert calls[1][2]['input'][-1]['call_id'] == 'call_test'
