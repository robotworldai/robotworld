import json
from concurrent.futures import ThreadPoolExecutor
from environment.runtime.events import EventLog


def test_concurrent_events_are_complete_ordered_and_immediately_readable(tmp_path):
    path = tmp_path / 'events/codex.jsonl'
    log = EventLog(path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: log.write('message', {'id': i, 'image': 'data:image/png;base64,AA=='}), range(100)))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r['sequence'] for r in records] == list(range(100))
    assert {r['payload']['id'] for r in records} == set(range(100))
    assert all(r['payload']['image'] == 'data:image/png;base64,AA==' for r in records)


def test_no_images_preserves_diagnostics_and_originals(tmp_path):
    from environment.runtime.events import export_without_images, without_images
    path = tmp_path / 'events/codex.jsonl'
    payload = {'params': {'content': [{'type': 'inputText', 'text': 'grasp missed'},
                                    {'type': 'inputImage', 'imageUrl': 'data:image/jpeg;base64,AA=='}]},
               'arguments': {'left_z': .8}, 'success': False}
    log = EventLog(path)
    log.write('tool_completed', payload)
    original = path.read_bytes()
    record = json.loads(original)
    target = path.parent / 'no-images/codex.jsonl'
    assert json.loads(target.read_text()) == without_images(record)
    assert 'data:image/' not in target.read_text()
    assert payload['params']['content'][1]['imageUrl'] == 'data:image/jpeg;base64,AA=='
    assert json.loads(target.read_text())['payload']['arguments'] == {'left_z': .8}
    target.unlink()
    export_without_images(path.parent)
    assert json.loads(target.read_text()) == without_images(record)
    assert path.read_bytes() == original
    # The exporter must not replace a valid analysis log if the source is incomplete.
    analysis = target.read_bytes()
    path.write_bytes(original + b'{')
    import pytest
    with pytest.raises(json.JSONDecodeError):
        export_without_images(path.parent)
    assert target.read_bytes() == analysis
