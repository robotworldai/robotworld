"""Shared observation window and explicit Responses request image selection."""
import base64
from collections import deque
import copy
import hashlib
import json
import re
from pathlib import Path


POLICY = "current-plus-4-history-stride-2-v1"
INSTRUCTIONS = (
    "Visual memory: each observation packet contains the current observation and up to four "
    "historical observations sampled every two observation rounds, not physics ticks. "
    "CURRENT and HISTORY labels include control-step timestamps. Only the latest packet's "
    "images remain in each model request; earlier image slots have explicit removal notices. "
    "Earlier text and action results remain. Use CURRENT images for the present scene."
)


def publish_image_window(output, parts, rounds):
    """Publish trusted packet metadata before handing the observation to Codex."""
    window = dict(policy=POLICY, image_history=4, history_interval=2, rounds=rounds,
                  image_sha256=[image_hash(p['imageUrl']) for p in parts if p['type']=='inputImage'])
    path = Path(output)/'image-window.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(window))
    temporary.replace(path)


def image_hash(url):
    if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
        raise ValueError("Visual history requires inline image data")
    return hashlib.sha256(base64.b64decode(url.split(";base64,", 1)[1], validate=True)).hexdigest()


class ObservationHistory:
    def __init__(self, image_history=4, history_interval=2):
        if type(image_history) is not int or type(history_interval) is not int or image_history < 0 or history_interval < 1:
            raise ValueError("Invalid observation history or interval")
        self.image_history = image_history
        self.history_interval = history_interval
        self.frames = deque(maxlen=image_history * history_interval + 1)
        self.included = []
        self.expected_hashes = []

    def append(self, sequence, env_step, image_parts):
        hashes = [image_hash(p["imageUrl"]) for p in image_parts if p["type"] == "inputImage"]
        if not hashes or (self.frames and sequence <= self.frames[-1][0]):
            raise ValueError("Observation history needs images and increasing round numbers")
        self.frames.append((sequence, env_step, image_parts, hashes))
        selected = {sequence - i * self.history_interval for i in range(self.image_history + 1)}
        parts, included, expected = [], [], []
        for number, step, images, digests in self.frames:
            if number not in selected:
                continue
            role = "CURRENT" if number == sequence else "HISTORY"
            parts.append(dict(type="inputText", text=f"{role} observation {number}; env step {step}"))
            parts.extend(images)
            included.append(dict(observation=number, env_step=step, role=role))
            expected.extend(digests)
        self.included, self.expected_hashes = included, expected
        return parts

    @property
    def instructions(self):
        return INSTRUCTIONS.replace("four", str(self.image_history)).replace("two observation rounds", f"{self.history_interval} observation rounds")

    def snapshot(self):
        return dict(policy=POLICY, image_history=self.image_history, history_interval=self.history_interval,
                    rounds=copy.deepcopy(self.included), image_sha256=list(self.expected_hashes))


def select_request_images(payload, window, *, allow_view_image=False):
    """Keep the trusted adapter packet, optionally windowing paired view_image results."""
    if payload.get("previous_response_id") or not isinstance(payload.get("input"), list):
        raise ValueError("Visual history requires explicit, stateless Responses input")
    expected = list(window["image_sha256"])
    if not expected or len(expected) > 50:
        raise ValueError("Observation packet is empty or exceeds the 50-image route limit")
    result = copy.deepcopy(payload)
    locations = []
    view_locations = []
    view_calls = set()
    for index, item in enumerate(result["input"]):
        if item.get("type") == "item_reference":
            raise ValueError("Remote item references cannot be image-audited")
        if (allow_view_image and item.get("type") == "function_call"
                and item.get("name") == "view_image" and item.get("call_id")
                and item.get("namespace") in (None, "functions")):
            view_calls.add(item["call_id"])
        for key in ("content", "output"):
            parts = item.get(key)
            if not isinstance(parts, list):
                continue
            images = [p for p in parts if p.get("type") == "input_image"]
            if images:
                is_view = (item.get("type") == "function_call_output" and key == "output"
                           and item.get("call_id") in view_calls)
                (view_locations if is_view else locations).append((index, key, images))
    if not locations:
        raise ValueError("Latest observation packet missing from request")
    def order(location):
        i, k, _ = location
        labels = [re.fullmatch(r"CURRENT observation ([0-9]+); env step ([0-9]+)", p.get('text', ''))
                  for p in result['input'][i][k] if p.get('type') == 'input_text']
        labels = [match for match in labels if match]
        if len(labels) != 1:
            raise ValueError('Latest observation packet differs from adapter history: stale or missing temporal labels')
        return (*map(int, labels[0].groups()), i)
    index, key, latest = max(locations, key=order)
    if [image_hash(p.get("image_url")) for p in latest] != expected:
        raise ValueError("Latest observation packet differs from adapter history")
    labels = [p.get("text") for p in result["input"][index][key] if p.get("type") == "input_text"]
    required = [f"{r['role']} observation {r['observation']}; env step {r['env_step']}" for r in window["rounds"]]
    if [label for label in labels if label in required] != required:
        raise ValueError("Latest observation packet has stale or missing temporal labels")
    # Crops are auxiliary tool results, never a replacement for the trusted packet.
    # Keep at most four newest images since that packet, within the route's total cap.
    extra_slots = [(i, k, n) for i, k, _ in view_locations if i > index
                   for n, p in enumerate(result["input"][i][k]) if p.get("type") == "input_image"]
    capacity = min(4, 50 - len(expected))
    if extra_slots and not capacity:
        raise ValueError("Observation packet leaves no room for view_image under the 50-image route limit")
    retained_extras = set(extra_slots[-capacity:]) if capacity else set()
    extra_hashes = [image_hash(result["input"][i][k][n].get("image_url"))
                    for i, k, n in extra_slots if (i, k, n) in retained_extras]
    removed = 0
    for old_index, old_key, images in locations:
        if (old_index, old_key) == (index, key):
            continue
        parts = result["input"][old_index][old_key]
        result["input"][old_index][old_key] = [
            dict(type="input_text", text="[Earlier packet image removed by visual-memory policy; "
                 "use the latest CURRENT/HISTORY packet. Original frame remains in the run archive.]")
            if p.get("type") == "input_image" else p for p in parts]
        removed += len(images)
    for i, k, _ in view_locations:
        for n, part in enumerate(result["input"][i][k]):
            if part.get("type") == "input_image" and (i, k, n) not in retained_extras:
                result["input"][i][k][n] = dict(type="input_text", text=
                    "[Earlier view_image removed by visual-memory policy; only the newest four "
                    "auxiliary images since the current observation packet are retained, within the 50-image cap.]")
                removed += 1
    return result, dict(**window, removed_image_occurrences=removed,
                        retained_input_index=index, retained_content_key=key,
                        auxiliary_image_policy="view-image-latest-4-since-packet-v1" if allow_view_image else "disabled",
                        auxiliary_image_sha256=extra_hashes,
                        outgoing_images=len(expected) + len(extra_hashes))
