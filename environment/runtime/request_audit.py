"""Fail-closed EEF audit with an optional, explicit observation-image window."""
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request
from environment.runtime.api_retry import open_with_retry
from environment.runtime.image_history import select_request_images


def summarize(value):
    result = {"input_images": 0, "image_sha256": [], "omission_notices": 0,
              "tools": [], "unsupported_tool_types": []}
    for tool in value.get("tools", []):
        if tool.get("type") == "function" and isinstance(tool.get("name"), str):
            result["tools"].append(tool["name"])
        else:
            result["unsupported_tool_types"].append(tool.get("type", "missing"))

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "input_image":
                result["input_images"] += 1
                url = node.get("image_url", "")
                if isinstance(url, str) and url.startswith("data:image/") and ";base64," in url:
                    data = base64.b64decode(url.split(";base64,", 1)[1], validate=True)
                    result["image_sha256"].append(hashlib.sha256(data).hexdigest())
            for key, item in node.items():
                if key not in ("image_url", "url"):
                    walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, str) and re.search(r"\bimage(?:s)?\s+(?:was |were |is |are )?omitted\b", node, re.I):
            result["omission_notices"] += 1
    walk(value.get("input", []))
    return result


def violations(record, allowed_tools, image_bytes=None):
    errors = []
    if allowed_tools is not None and (set(record["tools"]) != set(allowed_tools) or
            len(record["tools"]) != len(allowed_tools) or record["unsupported_tool_types"]):
        errors.append("tool_allowlist_mismatch")
    if image_bytes is not None and hashlib.sha256(image_bytes).hexdigest() not in record["image_sha256"]:
        errors.append("current_rgb_missing")
    return errors


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class RequestAudit:
    def __init__(self, config, output, allowed_tools, *, current_image=None, max_requests=110,
                 max_request_bytes=32 * 1024**2, max_retries=0,
                 image_window=None, allow_view_image=False):
        if type(max_retries) is not int or not 0 <= max_retries <= 3:
            raise ValueError("Upstream retries must be between zero and three")
        self.provider = config["model_provider"]
        self.upstream = config["model_providers"][self.provider]["base_url"].rstrip("/")
        self.output = Path(output)
        self.allowed_tools = None if allowed_tools is None else set(allowed_tools)
        self.current_image = current_image
        self.image_window = image_window
        self.allow_view_image = allow_view_image
        self.max_requests = max_requests
        self.max_request_bytes = max_request_bytes
        self.max_retries = max_retries
        self.records = []
        self.save_lock = threading.Lock()

    def controller_interrupt(self, reason):
        # Qualify only an in-flight final request, at the actual runner interrupt.
        if reason in {"timeout", "action_budget", "tool_budget", "give_up", "environment_end", "no_motion_guard"}:
            if self.records and not self.records[-1].get("finished"):
                self.records[-1]["controller_interrupt"] = reason
                self.records[-1]["controller_interrupt_time"] = time.time()

    @property
    def valid(self):
        def completed_or_cancelled(index, record):
            if record.get("terminal_event") == "response.completed":
                return True
            return (index == len(self.records)-1 and record.get("terminal_event") is None
                    and record.get("controller_interrupt") in
                    {"timeout", "action_budget", "tool_budget", "give_up", "environment_end"}
                    and record.get("client_disconnected") is True
                    and record.get("finished") is True)
        return bool(self.records) and all(not r.get("violations") and r.get("status") == 200
                                          and completed_or_cancelled(i, r)
                                          for i, r in enumerate(self.records))

    def save(self):
        with self.save_lock:
            temporary = self.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.records, indent=2) + "\n")
            temporary.replace(self.output)

    def __enter__(self):
        audit = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.path != "/v1/responses":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                record = {"violations": [], "status": None, "started_at": time.time()}
                audit.records.append(record)
                headers_sent = False
                try:
                    if not 0 < length < audit.max_request_bytes or len(audit.records) > audit.max_requests:
                        raise ValueError("request_budget")
                    body = self.rfile.read(length)
                    record.update(summarize(json.loads(body)))
                    record["route"] = "configured-provider"
                    current = audit.current_image() if audit.current_image else None
                    record["violations"] = violations(record, audit.allowed_tools, current)
                    audit.save()
                    if record["violations"]:
                        self.send_error(422, "EEF request boundary rejected")
                        return
                    if audit.image_window:
                        payload, selection = select_request_images(json.loads(body), audit.image_window(), allow_view_image=audit.allow_view_image)
                        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
                        record["image_window"] = selection
                    outgoing = summarize(json.loads(body))
                    if audit.image_window:
                        errors = violations(outgoing, audit.allowed_tools, current)
                        if outgoing["image_sha256"] != selection["image_sha256"] + selection.get("auxiliary_image_sha256", []) or outgoing["input_images"] != selection["outgoing_images"]:
                            errors.append("outgoing_image_window_mismatch")
                        record["violations"].extend(errors)
                        if errors:
                            raise ValueError("Outgoing visual-memory boundary rejected")
                    elif outgoing != {key: record[key] for key in outgoing}:
                        raise ValueError("Outgoing request changed image or tool contents")
                    record["outgoing_input_images"] = outgoing["input_images"]
                    record["outgoing_image_sha256"] = outgoing["image_sha256"]
                    record["outgoing_body_sha256"] = hashlib.sha256(body).hexdigest()
                    record["endpoint"] = audit.upstream + "/responses"
                    headers = {k: v for k, v in self.headers.items() if k.lower() not in
                               {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}}
                    headers["Accept-Encoding"] = "identity"
                    request = urllib.request.Request(audit.upstream + "/responses", data=body, headers=headers)
                    record["attempts"] = []
                    response = open_with_retry(urllib.request.build_opener(NoRedirect), request,
                        max_retries=audit.max_retries, timeout=90, attempts=record["attempts"],
                        persist=audit.save, cancelled=lambda: bool(record.get("controller_interrupt")))
                    with response:
                        record["status"] = response.status
                        audit.save()
                        self.send_response(response.status)
                        for key, value in response.headers.items():
                            if key.lower() not in {"connection", "transfer-encoding", "content-length"}:
                                self.send_header(key, value)
                        self.end_headers()
                        headers_sent = True
                        event_buffer = b""
                        while chunk := response.read1(65536):
                            event_buffer += chunk
                            while b"\n" in event_buffer:
                                line, event_buffer = event_buffer.split(b"\n", 1)
                                if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                                    event = json.loads(line[5:])
                                    kind = event.get("type", "")
                                    if kind in {"response.completed", "response.failed", "response.incomplete", "error"}:
                                        record["terminal_event"] = kind
                                        if kind != "response.completed":
                                            record["violations"].append(kind)
                            record["attempts"][-1]["response_forwarded"] = True
                            try:
                                self.wfile.write(chunk)
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError):
                                # This is downstream cancellation, not an upstream read failure.
                                record["client_disconnected"] = True
                                break
                except Exception as error:
                    record["violations"].append(type(error).__name__)
                    if not headers_sent:
                        try:
                            self.send_error(502, "EEF audit or upstream transport failed")
                        except (BrokenPipeError, ConnectionResetError):
                            record["client_disconnected"] = True
                    # Once streaming starts, never append a second HTTP response or retry.
                finally:
                    record["finished"] = True
                    record["finished_at"] = time.time()
                    record["retry_count"] = max(0, len(record.get("attempts", [])) - 1)
                    audit.save()

        class AuditServer(ThreadingHTTPServer):
            # Await the terminal event/disconnect before deciding score validity.
            daemon_threads = False
        self.server = AuditServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.overrides = [f'model_providers.{self.provider}.base_url="http://127.0.0.1:{self.server.server_port}/v1"']
        if self.max_retries:
            self.overrides += [f'model_providers.{self.provider}.stream_idle_timeout_ms=480000',
                               f'model_providers.{self.provider}.request_max_retries=0',
                               f'model_providers.{self.provider}.stream_max_retries=0']
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.save()
