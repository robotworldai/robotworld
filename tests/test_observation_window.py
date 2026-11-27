import copy
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from environment.runtime.observation_window import trim_observations, image_count
from environment.runtime.observation_window_proxy import ObservationWindowProxy

def packet(step, episode='e'):
    return {'type':'function_call_output','call_id':str(step),'output':[
        {'type':'input_text','text':json.dumps({'protocol':'robotworld.episode.v1','episode_id':episode,'control_step':step,'observation':{'state':step}})},
        *[{'type':'input_image','image_url':f'data:image/png;base64,{step}-{i}'} for i in range(15)]]}

def test_large_history_preserves_archive_and_text():
    body={'input':[packet(i) for i in range(25)]}
    original=copy.deepcopy(body)
    out,stats=trim_observations(body)
    assert body==original
    assert stats['images_before']==375 and stats['images_after']==15
    assert out['input'][-1]==body['input'][-1]
    for before,after in zip(body['input'],out['input']):
        assert before['call_id']==after['call_id']
        assert before['output'][0]==after['output'][0]

def test_duplicate_and_physical_order():
    out,stats=trim_observations({'input':[packet(10),packet(10),packet(8)]})
    assert stats['kept_control_step']==10
    assert image_count(out['input'][0])==0
    assert image_count(out['input'][1])==15
    assert image_count(out['input'][2])==0

@pytest.mark.parametrize('body',[
    {'previous_response_id':'hidden','input':[]},
    {'conversation':'hidden','input':[]},
    {'input':[packet(1),packet(2,'other')]},
    {'input':[{'type':'function_call_output','output':[{'type':'input_image','image_url':'x'}]}]},
    {'input':[{'type':'input_image','image_url':'x'}]*51},
])
def test_fail_closed(body):
    with pytest.raises(ValueError):trim_observations(body)

def test_actual_http_forward(tmp_path,monkeypatch):
    captured=[]
    class Upstream(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            captured.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
            self.wfile.write(b'data: {"ok":true}\n\n')
    server=ThreadingHTTPServer(('127.0.0.1',0),Upstream)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    monkeypatch.setenv('TEST_WINDOW_KEY','test-only')
    try:
        with ObservationWindowProxy(f'http://127.0.0.1:{server.server_port}/v1','TEST_WINDOW_KEY',tmp_path/'counts.jsonl') as relay:
            req=urllib.request.Request(relay.base_url+'/responses',data=json.dumps({'input':[packet(i) for i in range(25)]+auxiliary('http-view')}).encode(),headers={'Authorization':'Bearer test-only','Content-Type':'application/json'})
            with urllib.request.urlopen(req) as response:
                assert response.read()==b'data: {"ok":true}\n\n'
        assert image_count(captured[0])==16
        assert json.loads((tmp_path/'counts.jsonl').read_text())['images_before']==376
    finally:server.shutdown();server.server_close();thread.join()


def auxiliary(call_id, name='view_image', count=1):
    return [{'type':'function_call','call_id':call_id,'name':name,
             'arguments':json.dumps({'path':'/observations/crop.png'})},
            {'type':'function_call_output','call_id':call_id,'output':[
                {'type':'input_image','image_url':f'data:image/png;base64,aux-{call_id}-{i}'} for i in range(count)]}]


@pytest.mark.parametrize('name',['view_image','functions.view_image'])
def test_native_view_image_does_not_require_robot_metadata(name):
    body={'input':[packet(1),*auxiliary('view',name),packet(2)]}
    original=copy.deepcopy(body)
    out,stats=trim_observations(body)
    assert body==original
    assert stats['images_after']==16
    assert stats['kept_control_step']==2
    assert stats['auxiliary_images_after']==1
    assert out['input'][2]==body['input'][2]


def test_auxiliary_window_is_bounded_separately_from_robot_history():
    items=[packet(1)]
    for i in range(60):items.extend(auxiliary('v'+str(i)))
    items.append(packet(2))
    body={'input':items};original=copy.deepcopy(body)
    out,stats=trim_observations(body)
    assert body==original
    assert stats['images_after']==19
    assert stats['auxiliary_images_before']==60
    assert stats['auxiliary_images_after']==4
    assert out['input'][-1]==body['input'][-1]
    assert out['input'][-2]==body['input'][-2]


def test_auxiliary_cap_counts_images_not_calls():
    out,stats=trim_observations({'input':auxiliary('view',count=8)})
    assert image_count(out)==4
    assert stats['auxiliary_images_before']==8


def test_unknown_image_tool_is_not_silently_classified_as_view_image():
    with pytest.raises(ValueError,match='Unmarked'):
        trim_observations({'input':auxiliary('view','some_other_tool')})
