"""Offline provider regression: no API key, model endpoint or GPU required.

Run with the pinned provider interpreter to include the full wire roundtrip:
    PYTHONPATH=. /path/to/provider/bin/python -m unittest discover -s tests -p test_claude_provider_chat.py
"""
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest

from agents.run_claude import provider_proxy_config, configure_chat_provider


class ChatProviderConfigTests(unittest.TestCase):
    def test_chat_route_is_explicit_and_parameters_are_not_dropped(self):
        config = provider_proxy_config('deepseek-v4.1-flash')
        self.assertTrue(config['litellm_settings']['use_chat_completions_url_for_anthropic_messages'])
        self.assertFalse(config['litellm_settings']['drop_params'])
        self.assertEqual(config['model_list'][0]['litellm_params']['model'], 'openai/deepseek-v4.1-flash')
        self.assertEqual(config['model_list'][0]['litellm_params']['api_key'], 'os.environ/WORLD_MODEL_API_KEY')
        self.assertNotIn('supports_reasoning', json.dumps(config))

    @unittest.skipUnless(importlib.util.find_spec('litellm'), 'requires pinned provider dependencies')
    def test_messages_roundtrip_uses_chat_and_preserves_reasoning_image_and_tool(self):
        import litellm
        configure_chat_provider()
        from litellm.anthropic_interface.messages import create
        calls = []
        stream_mode = getattr(self, '_stream_mode', False)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                calls.append((self.path, body))
                if body.get('stream'):
                    chunks = [
                        {'role': 'assistant', 'reasoning_content': 'Offline fixture reasoning'},
                        {'tool_calls': [{'index': 0, 'id': 'call_probe', 'type': 'function',
                                         'function': {'name': 'execute', 'arguments': ''}}]},
                        {'tool_calls': [{'index': 0, 'function': {'arguments': '{"steps":1}'}}]},
                        {},
                    ]
                    payload = ''
                    for i, delta in enumerate(chunks):
                        payload += 'data: ' + json.dumps({
                            'id': 'offline-chat', 'object': 'chat.completion.chunk', 'created': 1,
                            'model': 'deepseek-v4.1-flash', 'choices': [{'index': 0,
                            'delta': delta, 'finish_reason': 'tool_calls' if i == 3 else None}],
                        }) + '\n\n'
                    payload += 'data: [DONE]\n\n'
                    data = payload.encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                data = json.dumps({
                    'id': 'offline-chat', 'object': 'chat.completion', 'created': 1,
                    'model': 'deepseek-v4.1-flash',
                    'choices': [{'index': 0, 'finish_reason': 'tool_calls', 'message': {
                        'role': 'assistant', 'content': None,
                        'tool_calls': [{'id': 'call_probe', 'type': 'function', 'function': {
                            'name': 'execute', 'arguments': '{"steps":1}'}}]}}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14},
                }).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        settings = provider_proxy_config('deepseek-v4.1-flash')['litellm_settings']
        old = {key: getattr(litellm, key, None) for key in settings}
        try:
            for key, value in settings.items():
                setattr(litellm, key, value)
            result = create(
                model='openai/deepseek-v4.1-flash',
                api_base=f'http://127.0.0.1:{server.server_port}/v1',
                api_key='offline-fake-key', max_tokens=128, stream=stream_mode,
                thinking={'type': 'enabled', 'budget_tokens': 4096},
                messages=[{'role': 'user', 'content': [
                    {'type': 'text', 'text': 'Observe and execute one step'},
                    {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'AA=='}}
                ]}],
                tools=[{'name': 'execute', 'description': 'Robot action', 'input_schema': {
                    'type': 'object', 'properties': {'steps': {'type': 'integer'}}, 'required': ['steps']}}],
                tool_choice={'type': 'auto'},
            )
            if stream_mode:
                events = list(result)
                raw = json.dumps(events, default=lambda value: value.decode() if isinstance(value, bytes) else value.model_dump())
                self.assertIn('execute', raw)
                self.assertIn('input_json_delta', raw)
                self.assertIn('thinking_delta', raw)
                self.assertIn('tool_use', raw)
            self.assertEqual(len(calls), 1)
            path, request = calls[0]
            self.assertEqual(path, '/v1/chat/completions')
            self.assertEqual(request['model'], 'deepseek-v4.1-flash')
            self.assertEqual(request['reasoning_effort'], 'high')
            self.assertEqual(request['tools'][0]['function']['name'], 'execute')
            self.assertEqual(request['messages'][0]['content'][1]['image_url']['url'], 'data:image/png;base64,AA==')
            if stream_mode:
                return
            if hasattr(result, 'model_dump'):
                result = result.model_dump()
            self.assertEqual(result['stop_reason'], 'tool_use')
            tool = next(block for block in result['content'] if block['type'] == 'tool_use')
            self.assertEqual(tool['name'], 'execute')
            self.assertEqual(tool['input'], {'steps': 1})
        finally:
            for key, value in old.items():
                setattr(litellm, key, value)
            server.shutdown()
            server.server_close()
            thread.join()

    def test_streaming_messages_roundtrip(self):
        self._stream_mode = True
        self.test_messages_roundtrip_uses_chat_and_preserves_reasoning_image_and_tool()


if __name__ == '__main__':
    unittest.main()
