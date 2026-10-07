"""Drive environment tools from Codex's agent loop, on the calling thread."""
import json
from pathlib import Path
import time

from .codex_session import CodexSession
from .events import EventLog
from .episode_recovery import TurnRecovery
from .nonaction_budget import attach, mode, NonActionBudgetExceeded


GIVE_UP = {"type": "function", "name": "give_up", "description": "End the episode because the task cannot be completed.",
           "inputSchema": {"type": "object", "properties": {"reason": {"type": "string"},
                           "hindsight": {"type": "string"}}, "required": ["reason", "hindsight"]}}


def run_episode(adapter, *, manifest, output_dir, model=None, config_overrides=(),
                instruction=None, max_actions=8, timeout_s=300, max_tools=None,
                on_interrupt=None, allow_give_up=True, continue_on_completion=False,
                idle_timeout_s=300, max_no_motion_calls=64, max_empty_turns=3,
                recover_failed_turns=False, nonaction_protocol=False):
    if (max_actions <= 0 or (timeout_s is not None and timeout_s <= 0)
            or (max_tools is not None and max_tools <= 0) or idle_timeout_s <= 0):
        raise ValueError("Budgets must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = {"action_calls": 0, "termination": None, "official_success": None,
              "runtime": "source-built-codex-app-server", "events": [],
              "tool_calls": 0, "model_turn_started": False}
    events = EventLog(output / "events/tools.jsonl")
    handlers = adapter.tool_handlers() if hasattr(adapter, "tool_handlers") else {"move_eef": adapter.move_eef}
    specs = adapter.tool_specs() if hasattr(adapter, "tool_specs") else [adapter.tool_spec()]
    if {s["name"] for s in specs} != set(handlers) or "give_up" in handlers:
        raise ValueError("Adapter tool declarations and handlers must match")
    read_only = getattr(adapter, "read_only_tools", set())
    developer = getattr(adapter, "developer_instructions",
                        "Use move_eef for robot motion. Each result includes fresh observations. "
                        "Never use shell, files, web or other tools to control the scene or read hidden state. "
                        "Execute one motion at a time and inspect its result. Do not claim official success yourself.")
    seen = {}
    signatures = {}
    deadline = time.monotonic() + timeout_s if timeout_s is not None else None
    adapter.deadline = deadline
    recovery = TurnRecovery(deadline, events) if recover_failed_turns and deadline is not None else None
    last_receipt = None
    no_motion_calls = empty_turns = 0
    turn_native_steps = 0
    if continue_on_completion:
        turn_native_steps = adapter.state["native_steps"]
        result.update(continuations=[], continue_on_completion=True)
    try:
        with CodexSession(manifest, output, config_overrides=config_overrides,
                          timeout=min(timeout_s, 180) if timeout_s is not None else 180) as session:
            if nonaction_protocol or mode() == 'observe':
                budget=attach(session,output,lambda:adapter.state['native_steps'])
            params = {"cwd": str(output.resolve()), "approvalPolicy": "never",
                      "baseInstructions": adapter.instructions,
                      "developerInstructions": developer,
                      "dynamicTools": [*specs, *([GIVE_UP] if allow_give_up else [])], "ephemeral": True,
                      "environments": [],
                      "config": {"features.shell_tool": False, "features.multi_agent": False,
                                 "features.code_mode": False, "features.apps": False,
                                 "features.plugins": False,
                                 "web_search": "disabled"}}
            (output / "prompt.json").write_text(json.dumps({
                "baseInstructions": params["baseInstructions"],
                "developerInstructions": params["developerInstructions"],
                "dynamicTools": params["dynamicTools"]}, indent=2) + "\n")
            if model:
                params["model"] = model
            if getattr(adapter, "require_isolated_thread", False):
                params.update(sandbox="read-only", runtimeWorkspaceRoots=[], selectedCapabilityRoots=[],
                              allowProviderModelFallback=False)
            started = session.rpc("thread/start", params)
            if getattr(adapter, "require_isolated_thread", False):
                boundary = {k: started.get(k) for k in
                            ("model", "instructionSources", "sandbox", "runtimeWorkspaceRoots")}
                (output / "thread-boundary.json").write_text(json.dumps(boundary, indent=2) + "\n")
                sandbox = boundary.get("sandbox") or {}
                if (boundary["instructionSources"] != [] or boundary["model"] != model or
                        boundary["runtimeWorkspaceRoots"] != [] or sandbox.get("type") != "readOnly" or
                        sandbox.get("networkAccess") is not False):
                    raise RuntimeError("Thread isolation or requested model was not preserved")
            thread = started["thread"]["id"]
            initial = adapter.observe_content()
            latest_content = list(initial)
            latest_observation = list(initial)
            initial.insert(0, {"type": "text", "text": instruction or adapter.task_instruction,
                               "text_elements": []})
            # Initial user input and dynamic tool outputs have different tags.
            initial = [{"type": "image", "url": x["imageUrl"]} if x["type"] == "inputImage"
                       else {"type": "text", "text": x["text"], "text_elements": []} for x in initial]
            turn = session.rpc("turn/start", {"threadId": thread, "input": initial})["turn"]["id"]
            result["model_turn_started"] = True
            while True:
                remaining = deadline - time.monotonic() if deadline is not None else idle_timeout_s
                if remaining <= 0:
                    result["termination"] = "timeout"
                    if on_interrupt:
                        on_interrupt(result["termination"])
                    session.request("turn/interrupt", {"threadId": thread, "turnId": turn})
                    break
                try:
                    message = session.receive(remaining)
                except TimeoutError:
                    result["termination"] = "timeout" if deadline is not None else "model_idle_timeout"
                    if on_interrupt:
                        on_interrupt(result["termination"])
                    session.request("turn/interrupt", {"threadId": thread, "turnId": turn})
                    break
                method = message.get("method")
                if method == "item/tool/call":
                    call = message["params"]
                    call_id = call["callId"]
                    signature = json.dumps([call.get("threadId"), call.get("turnId"),
                                            call["tool"], call["arguments"]], sort_keys=True)
                    if call_id in seen:
                        if signatures[call_id] != signature:
                            raise RuntimeError("A tool call ID was reused with different arguments")
                        session.send({"id": message["id"], "result": seen[call_id]})
                        continue
                    if call.get("threadId") != thread or call.get("turnId") != turn:
                        raise RuntimeError("Tool call belongs to another thread or turn")
                    name = call["tool"]
                    events.write("tool_requested", call)
                    exhausted = max_tools is not None and result["tool_calls"] >= max_tools
                    if not exhausted:
                        result["tool_calls"] += 1
                    if exhausted:
                        result["termination"] = "tool_budget"
                        response = {"success": False, "contentItems": [{"type": "inputText", "text": "Tool budget exhausted; no command executed."}]}
                    elif name == "give_up" and allow_give_up:
                        result["give_up_arguments"] = call["arguments"]
                        result["events"].append({"call_id": call_id, "tool": name, "arguments": call["arguments"]})
                        feedback = "Episode stopped by give_up."
                        response = {"success": True, "contentItems": [{"type": "inputText", "text": feedback}]}
                        result["termination"] = "give_up"
                    elif name in handlers and not adapter.ended():
                        if name not in read_only and result["action_calls"] >= max_actions:
                            result["termination"] = "action_budget"
                            response = {"success": False, "contentItems": [{"type": "inputText", "text": "Action budget exhausted; no motion executed."}]}
                        else:
                            if name not in read_only:
                                result["action_calls"] += 1
                            response, trace = handlers[name](call["arguments"])
                            result["events"].append({"call_id": call_id, "tool": name,
                                                      "arguments": call["arguments"], "execution": trace})
                            if continue_on_completion and not nonaction_protocol:
                                no_motion_calls = 0 if trace.get("executed_native_steps", 0) else no_motion_calls + 1
                                if no_motion_calls >= max_no_motion_calls:
                                    result["termination"] = "no_motion_guard"
                            if adapter.ended():
                                result["termination"] = "environment_end"
                    else:
                        response = {"success": False, "contentItems": [{"type": "inputText", "text": "Tool unavailable or episode already ended."}]}
                        if continue_on_completion and not nonaction_protocol:
                            no_motion_calls += 1
                            if no_motion_calls >= max_no_motion_calls:
                                result["termination"] = "no_motion_guard"
                    events.write("tool_completed", {"call_id": call_id, "tool": name, "response": response})
                    seen[call_id] = response
                    signatures[call_id] = signature
                    latest_content = response['contentItems']
                    if any(x.get('type') == 'inputImage' for x in latest_content):
                        latest_observation = list(latest_content)
                    last_receipt = dict(call_id=call_id, tool=name,
                                        action_calls=result['action_calls'], tool_calls=result['tool_calls'])
                    session.send({"id": message["id"], "result": response})
                    (output / "episode.json").write_text(json.dumps(result, indent=2) + "\n")
                    if result["termination"]:
                        if on_interrupt:
                            on_interrupt(result["termination"])
                        session.request("turn/interrupt", {"threadId": thread, "turnId": turn})
                        break
                elif method == "turn/completed":
                    completed = message["params"]["turn"]
                    if recovery is not None and completed['status'] == 'failed':
                        if (message['params'].get('threadId') != thread or completed.get('id') != turn):
                            raise RuntimeError('Failed turn belongs to another thread or turn')
                        if not adapter.ended() and recovery.wait(completed.get('error'),
                                thread=thread, turn=turn,
                                control_step=getattr(adapter, 'state', {}).get('native_steps')):
                            # Reuse already published feedback. No new observation, physics tick or action replay.
                            initial = [{'type': 'image', 'url': x['imageUrl']} if x['type'] == 'inputImage'
                                       else {'type': 'text', 'text': x['text'], 'text_elements': []}
                                       for x in latest_content]
                            initial.insert(0, {'type': 'text', 'text':
                                'The previous model turn failed at the model service. '
                                'Continue the same physical episode from the supplied observation and feedback. '
                                'Do not replay completed actions. Last completed tool receipt: '+json.dumps(last_receipt),
                                'text_elements': []})
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise TimeoutError('Agent wall timeout before recovery turn')
                            turn = session.rpc('turn/start', {'threadId': thread, 'input': initial},
                                               timeout=remaining)['turn']['id']
                            result['recoveries'] = recovery.attempts
                            continue
                    if continue_on_completion and completed["status"] == "completed":
                        if adapter.ended():
                            result["termination"] = "environment_end"
                            break
                        now_steps = adapter.state["native_steps"]
                        empty_turns = empty_turns + 1 if now_steps == turn_native_steps else 0
                        if not nonaction_protocol and empty_turns >= max_empty_turns:
                            result["termination"] = "agent_no_action"
                            break
                        if max_tools is not None and result["tool_calls"] >= max_tools:
                            result["termination"] = "tool_budget"
                            break
                        if result["action_calls"] >= max_actions:
                            result["termination"] = "action_budget"
                            break
                        if deadline is not None and time.monotonic() >= deadline:
                            result["termination"] = "timeout"
                            break
                        # Same Codex thread and physical episode, no reset or privileged hint.
                        message = ("The task is not complete and the control-step budget remains. "
                                   "Continue taking meaningful bounded actions using the current RGB and proprioception. "
                                   "Do not stop only because the task looks difficult or objects are not yet located. "
                                   "Change your exploration or manipulation strategy based on observed feedback.")
                        # Reuse delivered sensors; text-only turns do not create a new sample.
                        initial = list(latest_observation)
                        initial = [{"type": "image", "url": x["imageUrl"]} if x["type"] == "inputImage"
                                   else {"type": "text", "text": x["text"], "text_elements": []} for x in initial]
                        initial.insert(0, {"type": "text", "text": message, "text_elements": []})
                        result["continuations"].append(dict(after_native_steps=now_steps,
                                                            after_actions=result["action_calls"],
                                                            empty_turns=empty_turns, instruction=message))
                        (output / "episode.json").write_text(json.dumps(result, indent=2) + "\n")
                        turn_native_steps = now_steps
                        turn = session.rpc("turn/start", {"threadId": thread, "input": initial})["turn"]["id"]
                        continue
                    result["termination"] = "agent_" + completed["status"]
                    if completed.get("error"):
                        result["error"] = completed["error"]
                    break
                elif method == "item/completed":
                    item = message.get("params", {}).get("item", {})
                    if item.get("type") == "agentMessage":
                        result.setdefault("agent_messages", []).append(item.get("text", ""))
                elif method == "thread/tokenUsage/updated":
                    result["token_usage"] = message.get("params", {}).get("tokenUsage")
                elif "id" in message and method:
                    session.send({"id": message["id"], "error": {"code": -32601,
                                 "message": "Only robot dynamic tool requests are supported"}})
    except NonActionBudgetExceeded as stop:
        result['termination']=stop.snapshot['stop_reason']
        result['interaction_budget']=stop.snapshot
    except BaseException as error:
        result["termination"] = "infrastructure_error"
        result["error_type"] = type(error).__name__
        raise
    finally:
        if 'budget' in locals() and budget is not None:
            budget.progress()
            result['interaction_budget']=budget.snapshot()
        try:
            result["official_success"] = adapter.official_success()
        except Exception as error:
            result["status_error_type"] = type(error).__name__
        (output / "episode.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
