"""JSON-RPC host for source-built Codex. No model HTTP client."""
import hashlib
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import threading
import time
from collections import deque
from .events import EventLog


class CodexSession:
    def __init__(self, manifest, output_dir, *, config_overrides=(), timeout=180):
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.manifest = self.verify_build(manifest)
        self.timeout = timeout
        self.overrides = list(config_overrides)
        self.messages = queue.Queue()
        self.counter = 0
        self.process = None
        self.pending = deque()
        self.events = EventLog(self.output / "events/codex.jsonl")

    @staticmethod
    def verify_build(path):
        manifest = json.loads(Path(path).read_text())
        expected = Path(__file__).resolve().parents[2] / "codex"
        if Path(manifest["source"]).resolve() != expected:
            raise ValueError("Only the World/codex source checkout is allowed")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=expected, text=True).strip()
        if commit != manifest["commit"]:
            raise ValueError("Source revision differs from the binary build record")
        patch = manifest.get('local_patch')
        if patch:
            from .patched_codex_build import source_fingerprint
            if patch.get('id') != 'robot-image-window-v1' or source_fingerprint(expected) != patch['sha256']:
                raise ValueError('Codex local patch differs from the build record')
        elif subprocess.check_output(["git", "status", "--porcelain"], cwd=expected):
            raise ValueError("Codex worktree is not clean")
        binary = Path(manifest["binary"]).resolve()
        build_name = commit if not patch else commit+'-image-window-work'
        build_root = (expected.parent / "var/build/codex" / build_name).resolve()
        if not binary.is_relative_to(build_root):
            raise ValueError("Binary must be an artifact of the local source build")
        if hashlib.sha256(binary.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError("Codex binary checksum mismatch")
        for helper, digest in manifest.get("helpers", {}).items():
            helper = Path(helper).resolve()
            if not helper.is_relative_to(build_root) or hashlib.sha256(helper.read_bytes()).hexdigest() != digest:
                raise ValueError("Codex helper is not a verified local build artifact")
        return manifest

    def __enter__(self):
        self.stderr = (self.output / "codex.stderr.log").open("w")
        self.connection = None
        endpoint = os.environ.get('WORLD_CODEX_SOCKET')
        if endpoint:
            self.connection = socket.socket(socket.AF_UNIX)
            self.connection.connect(endpoint)
            self.input_stream = self.connection.makefile('w', encoding='utf-8')
            self.output_stream = self.connection.makefile('r', encoding='utf-8')
        else:
            self._spawn_local()
            self.input_stream = self.process.stdin
            self.output_stream = self.process.stdout
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            self.rpc("initialize", {"clientInfo": {"name": "world_robot_eval", "version": "0.1"},
                                    "capabilities": {"experimentalApi": True}})
            self.send({"method": "initialized"})
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _spawn_local(self):
        command = [self.manifest["binary"], "--listen", "stdio://"]
        for override in self.overrides:
            command.extend(["-c", override])
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, text=True, bufsize=1,
                                        cwd=self.output, env=os.environ.copy())

    def _read(self):
        try:
            for line in self.output_stream:
                try:
                    message = json.loads(line)
                    self.events.write("server_to_client", message)
                    self.messages.put(message)
                except json.JSONDecodeError:
                    self.messages.put(RuntimeError("Non-JSON output from Codex app-server"))
        finally:
            self.messages.put(EOFError("Codex app-server closed stdout; inspect codex.stderr.log"))

    def send(self, message):
        self.events.write("client_to_server", message)
        self.input_stream.write(json.dumps(message, allow_nan=False) + "\n")
        self.input_stream.flush()
        budget = getattr(self, 'nonaction_budget', None)
        if budget is not None:
            budget.reply(message)

    def request(self, method, params):
        if (method == 'thread/start' and hasattr(self, 'nonaction_budget')
                and not self.nonaction_budget.observe_only):
            from .nonaction_budget import instructions
            params = dict(params, developerInstructions=params.get('developerInstructions','')+'\n'+instructions(self.nonaction_budget))
        if method == 'turn/start' and hasattr(self, 'nonaction_budget'):
            from .nonaction_budget import check
            check(self)
            self._budget_thread = params['threadId']
        self.counter += 1
        self.send({"id": self.counter, "method": method, "params": params})
        return self.counter

    def receive(self, timeout=None):
        from .nonaction_budget import check
        check(self)
        if self.pending:
            message = self.pending.popleft()
        else:
            message = self._receive_wire(timeout)
        budget = getattr(self, 'nonaction_budget', None)
        if budget is not None:
            budget.observe(message)
            if message.get('method') != 'item/tool/call':
                check(self)
        return message

    def _receive_wire(self, timeout=None):
        try:
            item = self.messages.get(timeout=self.timeout if timeout is None else timeout)
        except queue.Empty as error:
            raise TimeoutError("Timed out waiting for Codex app-server") from error
        if isinstance(item, BaseException):
            raise item
        return item

    def rpc(self, method, params, *, timeout=None):
        request_id = self.request(method, params)
        deadline = time.monotonic() + (self.timeout if timeout is None else min(self.timeout, timeout))
        while True:
            message = self._receive_wire(max(0, deadline - time.monotonic()))
            if message.get("id") == request_id and "method" not in message:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                if method=='turn/start' and hasattr(self,'nonaction_budget'):
                    self.nonaction_budget.start_turn(message['result']['turn']['id'])
                return message["result"]
            # Tool requests may race the turn/start response. Preserve them for
            # the episode loop instead of rejecting or losing an actual action.
            self.pending.append(message)

    def __exit__(self, *_):
        if hasattr(self,'nonaction_budget'):
            self.nonaction_budget.progress()
            self.nonaction_budget.write()
        if getattr(self, 'connection', None):
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.reader.join(timeout=2)
            self.input_stream.close()
            self.output_stream.close()
            self.connection.close()
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            if hasattr(self, "reader"):
                self.reader.join(timeout=2)
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    stream.close()
        self.stderr.close()
