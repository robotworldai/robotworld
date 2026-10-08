import base64
import copy
import io
import json
import urllib.request

import pytest

from environment.runtime.image_history import ObservationHistory, select_request_images
from environment.runtime.request_audit import summarize


def image(data):
    return dict(type="inputImage", imageUrl="data:image/png;base64," + base64.b64encode(data).decode())


def wire(parts):
    return [dict(type="input_image", image_url=p["imageUrl"]) if p["type"] == "inputImage"
            else dict(type="input_text", text=p["text"]) for p in parts]


@pytest.mark.parametrize("cameras", [1, 2, 3])
def test_more_than_fifty_rounds_keep_only_current_and_four_spaced_observations(cameras):
    history = ObservationHistory()
    payload = dict(model="fixture", input=[], tools=[dict(type="function", name="step")])
    for number in range(65):
        parts = history.append(number, number * 15, [image(f"{number}/{c}".encode()) for c in range(cameras)])
        if number:
            payload["input"].extend([
                dict(type="function_call", name="step", call_id=str(number), arguments="{}"),
                dict(type="function_call_output", call_id=str(number), output=wire(parts))])
        else:
            payload["input"].append(dict(role="user", content=wire(parts)))
        original = copy.deepcopy(payload)
        outgoing, report = select_request_images(payload, history.snapshot())
        assert payload == original
        assert summarize(outgoing)["input_images"] == min(5, number // 2 + 1) * cameras
        assert summarize(outgoing)["image_sha256"] == history.expected_hashes
        assert outgoing["input"][-1] == payload["input"][-1]
        for before, after in zip(payload["input"][:-1], outgoing["input"][:-1], strict=True):
            key = "output" if "output" in before else "content"
            if key not in before:
                assert before == after
                continue
            assert {k:v for k,v in before.items() if k != key} == {k:v for k,v in after.items() if k != key}
            for a, b in zip(before[key], after[key], strict=True):
                if a["type"] != "input_image":
                    assert a == b
                else:
                    assert b["type"] == "input_text" and "Earlier packet image removed" in b["text"]
    assert [r["observation"] for r in report["rounds"]] == [56, 58, 60, 62, 64]
    assert report["removed_image_occurrences"] > 50
    assert len(history.frames) == 9


def test_repeated_identical_pixels_keep_temporal_identity_and_empty_tool_outputs():
    history = ObservationHistory()
    payload = dict(input=[])
    for number in range(12):
        parts = history.append(number, 0, [image(b"same pixels")])
        payload["input"].append(dict(type="function_call_output", call_id=str(number), output=wire(parts)))
    payload["input"].append(dict(type="function_call_output", call_id="invalid", output="No action executed"))
    outgoing, report = select_request_images(payload, history.snapshot())
    assert report["outgoing_images"] == 5
    assert [r["observation"] for r in report["rounds"]] == [3, 5, 7, 9, 11]
    assert outgoing["input"][-1] == payload["input"][-1]
    again, _ = select_request_images(outgoing, history.snapshot())
    assert again == outgoing
    payload["input"].pop(-2)
    with pytest.raises(ValueError, match="temporal labels"):
        select_request_images(payload, history.snapshot())


def test_nondefault_history_remains_explicit_and_provider_cap_is_not_silent():
    history = ObservationHistory(0, 1)
    assert "up to 0 historical" in history.instructions
    assert "every 1 observation rounds" in history.instructions
    parts = history.append(0, 0, [image(str(i).encode()) for i in range(51)])
    with pytest.raises(ValueError, match="50-image"):
        select_request_images(dict(input=[dict(role="user", content=wire(parts))]), history.snapshot())


@pytest.mark.parametrize("change", ["missing_latest", "missing_history", "wrong_order", "previous_response", "reference"])
def test_missing_or_changed_packet_fails_closed(change):
    history = ObservationHistory()
    payload = dict(input=[])
    for number in range(5):
        parts = history.append(number, number, [image(str(number).encode())])
        payload["input"].append(dict(role="user", content=wire(parts)))
    if change == "missing_latest":
        payload["input"].pop()
    elif change == "missing_history":
        payload["input"][-1]["content"].pop(1)
    elif change == "wrong_order":
        payload["input"][-1]["content"].reverse()
    elif change == "previous_response":
        payload["previous_response_id"] = "opaque"
    else:
        payload["input"].insert(0, dict(type="item_reference", id="opaque"))
    with pytest.raises(ValueError):
        select_request_images(payload, history.snapshot())


@pytest.mark.parametrize("dynamic_tools", [False, True])
def test_actual_http_audit_trims_history_before_retry_without_changing_current(tmp_path, monkeypatch, dynamic_tools):
    from environment.runtime import request_audit as module
    history, payload = ObservationHistory(), dict(model="fixture-model", tools=[dict(type="function", name="step")], input=[])
    for number in range(65):
        parts = history.append(number, number * 15, [image(str(number).encode())])
        payload["input"].append(dict(type="function_call_output", call_id=str(number), output=wire(parts)))
    if dynamic_tools:
        add_view_image(payload, "crop-current")
    seen = []
    class Response(io.BytesIO):
        status = 200
        headers = {}
    class Opener:
        def open(self, request, **kwargs):
            seen.append(request.data)
            if len(seen) == 1:
                raise TimeoutError()
            return Response(b'data: {"type":"response.completed"}\n\n')
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *args: Opener())
    original = module.open_with_retry
    monkeypatch.setattr(module, "open_with_retry", lambda *args, **kwargs: original(*args, **kwargs, sleep=lambda _: None))
    cfg = dict(model_provider="fixture", model_providers={"fixture": dict(base_url="http://invalid")})
    with module.RequestAudit(cfg, tmp_path / "audit.json", None if dynamic_tools else {"step"}, current_image=lambda: b"64",
                             max_retries=3, image_window=history.snapshot, allow_view_image=dynamic_tools) as audit:
        req = urllib.request.Request(f"http://127.0.0.1:{audit.server.server_port}/v1/responses", data=json.dumps(payload).encode())
        with client.open(req) as response:
            response.read()
    assert audit.valid and len(seen) == 2 and seen[0] == seen[1]
    assert summarize(json.loads(seen[0]))["input_images"] == 5 + int(dynamic_tools)
    record = audit.records[0]
    assert record["input_images"] > 50 and record["outgoing_input_images"] == 5 + int(dynamic_tools)
    assert record["image_window"]["rounds"][-1]["observation"] == 64
    assert record["retry_count"] == 1


def add_view_image(payload, call_id, data=b"cropped keyboard"):
    payload["input"].extend([
        dict(type="function_call", name="view_image", namespace="functions",
             call_id=call_id, arguments='{"path":"/workspace/keyboard.png"}'),
        dict(type="function_call_output", call_id=call_id, output=wire([image(data)]))])


def test_view_image_after_latest_packet_preserves_crop_and_tool_history():
    history = ObservationHistory()
    payload = dict(input=[])
    for number in range(4):
        parts = history.append(number, number * 15, [image(f"{number}/{c}".encode()) for c in range(5)])
        payload["input"].append(dict(role="user", content=wire(parts)))
    add_view_image(payload, "crop")
    original = copy.deepcopy(payload)
    # The old strict route reproduces task 41's failure; opt-in fixes only the builtin route.
    with pytest.raises(ValueError, match="differs from adapter"):
        select_request_images(payload, history.snapshot())
    outgoing, report = select_request_images(payload, history.snapshot(), allow_view_image=True)
    assert payload == original
    assert outgoing["input"][-3:] == payload["input"][-3:]
    assert summarize(outgoing)["input_images"] == report["outgoing_images"] == 11
    assert summarize(outgoing)["image_sha256"] == history.expected_hashes + report["auxiliary_image_sha256"]
    again, _ = select_request_images(outgoing, history.snapshot(), allow_view_image=True)
    assert again == outgoing


@pytest.mark.parametrize("cameras,retained", [(5, 4), (49, 1)])
def test_auxiliary_window_keeps_newest_without_displacing_current(cameras, retained):
    history = ObservationHistory()
    initial = history.append(0, 0, [image(b"old")])
    payload = dict(input=[dict(role="user", content=wire(initial))])
    add_view_image(payload, "old-crop")
    current = history.append(1, 10, [image(str(i).encode()) for i in range(cameras)])
    payload["input"].append(dict(role="user", content=wire(current)))
    for i in range(6):
        add_view_image(payload, str(i), str(i).encode())
    outgoing, report = select_request_images(payload, history.snapshot(), allow_view_image=True)
    assert outgoing["input"][3] == payload["input"][3]
    assert outgoing["input"][-retained*2:] == payload["input"][-retained*2:]
    assert summarize(outgoing)["input_images"] == cameras + retained <= 50
    assert report["outgoing_images"] == cameras + retained
    for before, after in zip(payload["input"], outgoing["input"], strict=True):
        if before.get("type") == "function_call":
            assert before == after
    assert outgoing["input"][2]["output"][0]["type"] == "input_text"


@pytest.mark.parametrize("change", ["missing_packet", "changed_packet", "missing_label", "unpaired", "wrong_tool"])
def test_auxiliary_images_never_bypass_observation_validation(change):
    history = ObservationHistory()
    packet = history.append(0, 0, [image(b"trusted")])
    payload = dict(input=[dict(role="user", content=wire(packet))])
    add_view_image(payload, "crop")
    if change == "missing_packet":
        payload["input"].pop(0)
    elif change == "changed_packet":
        payload["input"][0]["content"][-1] = wire([image(b"wrong")])[0]
    elif change == "missing_label":
        payload["input"][0]["content"].pop(0)
    elif change == "unpaired":
        payload["input"].pop(1)
    else:
        payload["input"][1]["name"] = "unknown_tool"
    with pytest.raises(ValueError):
        select_request_images(payload, history.snapshot(), allow_view_image=True)


def test_full_observation_packet_does_not_silently_drop_latest_crop():
    history = ObservationHistory()
    packet = history.append(0, 0, [image(str(i).encode()) for i in range(50)])
    payload = dict(input=[dict(role="user", content=wire(packet))])
    add_view_image(payload, "crop")
    with pytest.raises(ValueError, match="no room for view_image"):
        select_request_images(payload, history.snapshot(), allow_view_image=True)
