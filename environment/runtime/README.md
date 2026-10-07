# Codex episode runtime

`codex_session.py` runs the verified local source-built app-server; `episode_runner.py` dispatches its dynamic tools. No direct model HTTP client is used.

Each new run stores append-only, flushed JSONL evidence under `events/`:

- `codex.jsonl`: both directions of app-server JSON-RPC, including initial instructions, image payloads, notifications, tool calls, responses and model usage. Records all events exposed by app-server; does not claim access to undisclosed model internals.
- `tools.jsonl`: requested/completed tools, including give_up and rejected requests. An unmatched request indicates incomplete execution; do not blindly replay it.
- `environment.jsonl`: observation state and frame references, each native action before dispatch and measured robot state after execution. Images for every control step are in continuous videos; model observation PNGs and history indices are in `frames/`.

`episode.json` is the outcome summary. Event records have timestamps and sequence numbers within each file. Raw image payloads are retained, so event logs can be large. These are runtime artifacts under `var/`, not part of the GitHub source bundle.

The RoboDojo launcher accepts `--max-env-steps`; `evaluation-config.json` records both original and effective limits. A changed budget is a diagnostic evaluation and must not be reported as the standard benchmark setting.

## No image analysis version

Follow-up on automatic sync generation of `events/no-images/{codex,tools,environment}.jsonl`.
Keeps the serial number, time, message, tool parameter, execution feedback and state corresponding to the original event, replacing only the embedded image data URL with a placeholder with MIME, SHA256 and the length of the original string. The same image has the same Hash, in the same order; Picture path, etc.

This copy is suitable for retrieval and error analysis and is not the protocol data that can be sent directly to the model. The complete image is still in the original event and `frames/`, and the continuous execution of the image is in the video.

Back-up generation for closed old operations (no changes to original records):

```bash
python3 environment/runtime/events.py var/runs/docker/isaac601-deposit-coin/task02-1000steps/events
```

Export commands only for closed operations; Runs two records by EventLog.
