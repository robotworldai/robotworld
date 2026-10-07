"""Observation chronology must not depend on tool receipt list order."""
import base64
import copy

import pytest

from environment.runtime.image_history import ObservationHistory, select_request_images


def packet(history, step):
    image = dict(type="inputImage", imageUrl="data:image/png;base64," +
                 base64.b64encode(str(step).encode()).decode())
    parts = history.append(step, step, [image])
    return [dict(type="input_image", image_url=p["imageUrl"])
            if p["type"] == "inputImage" else dict(type="input_text", text=p["text"])
            for p in parts]


def test_reversed_receipts_keep_latest_observation_without_reordering_actions():
    history = ObservationHistory()
    old, new = packet(history, 31), packet(history, 35)
    payload = dict(input=[
        dict(type="function_call_output", call_id="new", output=new),
        dict(type="function_call_output", call_id="old", output=old),
    ])
    original = copy.deepcopy(payload)
    result, report = select_request_images(payload, history.snapshot())
    assert payload == original
    assert result["input"][0] == original["input"][0]
    assert result["input"][1]["call_id"] == "old"
    assert not any(p["type"] == "input_image" for p in result["input"][1]["output"])
    assert report["retained_input_index"] == 0
    assert select_request_images(result, history.snapshot())[0] == result


def test_already_removed_latest_images_cannot_fall_back_to_stale_packet():
    history = ObservationHistory()
    old, new = packet(history, 31), packet(history, 35)
    new = [dict(type="input_text", text="[Earlier packet image removed]")
           if p["type"] == "input_image" else p for p in new]
    payload = dict(input=[
        dict(type="function_call_output", call_id="new", output=new),
        dict(type="function_call_output", call_id="old", output=old),
    ])
    with pytest.raises(ValueError, match="differs from adapter history"):
        select_request_images(payload, history.snapshot())
