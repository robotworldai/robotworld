import copy
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from environment.runtime.messages_window_proxy import MessagesWindowProxy, image_count, trim_messages


def pair(step, identity=None, name='mcp__robotworld__observe', episode='e'):
    identity = str(step) if identity is None else identity
    metadata = dict(protocol='robotworld.episode.v1', episode_id=episode, control_step=step,
                    observation_id=f'{episode}:{step}', observation={'state':{}})
    return [dict(role='assistant',content=[dict(type='tool_use', id=identity, name=name, input={})]),
            dict(role='user',content=[dict(type='tool_result', tool_use_id=identity, content=[
                dict(type='text',text=json.dumps(metadata)),
                *[dict(type='image',source=dict(type='base64',media_type='image/jpeg',data='test')) for _ in range(15)]])])]


def test_complete_history_trims_only_recognized_robotworld_images():
    body={'model':'test','messages':sum([pair(i) for i in range(25)],[])}
    body['messages'] += pair(26,name='another_tool')
    original=copy.deepcopy(body)
    output, stats=trim_messages(body)
    assert body==original
    assert stats['images_before']==390 and stats['images_after']==30
    assert output['messages'][-2:]==body['messages'][-2:]
    for i in range(24):
        before=body['messages'][2*i+1]['content'][0]
        after=output['messages'][2*i+1]['content'][0]
        assert before['tool_use_id']==after['tool_use_id']
        assert before['content'][0]==after['content'][0]
        assert not image_count(after)


def test_physical_step_duplicate_and_execute_without_image():
    body={'messages':pair(10,'first')+pair(10,'second')+pair(8,'older')}
    body['messages'] += [dict(role='assistant', content=[dict(type='tool_use',id='act',name='mcp__robotworld__execute',input={})]),
                         dict(role='user',content=[dict(type='tool_result',tool_use_id='act',content=[dict(type='text',text='{}')])])]
    output,stats=trim_messages(body)
    assert stats['kept_control_step']==10 and stats['kept_tool_use_id']=='second'
    assert image_count(output['messages'][1])==0
    assert image_count(output['messages'][3])==15
    assert image_count(output['messages'][5])==0
    assert output['messages'][-2:]==body['messages'][-2:]


def test_unmarked_or_nonrobot_images_never_silently_removed():
    body={'messages':sum([pair(i,name='nonrobot') for i in range(10)],[])}
    output,stats=trim_messages(body)
    assert output==body and stats['images_after']==150
    with pytest.raises(ValueError,match='explicitly configured'):
        trim_messages(body,max_images=50)
    malformed={'messages':pair(1)}
    malformed['messages'][1]['content'][0]['content'][0]['text']='{}'
    with pytest.raises(ValueError,match='metadata'):trim_messages(malformed)
    with pytest.raises(ValueError,match='Mixed'):trim_messages({'messages':pair(1)+pair(2,episode='other')})
    # Compacted/unpaired history stays intact; no tool identity means no authority to trim.
    unpaired={'messages':[pair(3)[1]]}
    assert trim_messages(unpaired)[0]==unpaired


def test_http_messages_and_count_tokens_streaming(tmp_path):
    captured=[]
    class Upstream(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            captured.append((self.path,json.loads(self.rfile.read(int(self.headers['Content-Length']))),self.headers.get('anthropic-version')))
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
            self.wfile.write(b'event: message_stop\ndata: {"type":"message_stop"}\n\n')
    server=ThreadingHTTPServer(('127.0.0.1',0),Upstream)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with MessagesWindowProxy(f'http://127.0.0.1:{server.server_port}/v1','test-key',tmp_path/'counts.jsonl') as proxy:
            for path in ('/v1/messages?beta=true','/v1/messages/count_tokens'):
                request=urllib.request.Request(proxy.base_url+path,data=json.dumps({'messages':pair(0)+pair(2)}).encode(),
                    headers={'x-api-key':'test-key','anthropic-version':'2023-06-01','Content-Type':'application/json'})
                with urllib.request.urlopen(request) as response:
                    assert response.read()==b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        assert [row[0] for row in captured]==['/v1/messages?beta=true','/v1/messages/count_tokens']
        assert all(image_count(body)==15 and version=='2023-06-01' for _,body,version in captured)
        stats=[json.loads(line) for line in (tmp_path/'counts.jsonl').read_text().splitlines()]
        assert len(stats)==2 and all(s['images_before']==30 and s['images_after']==15 for s in stats)
    finally:server.shutdown();server.server_close();thread.join()
