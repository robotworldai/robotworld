# Fixed Scripted 1v1 Opponent

Ported from RoboWorld_Scaffold commit
`34689f4a449ee0ede5c17bee7afae1263a4db059`
(`upload/upload-github-20261001`). The two opponent source files are unchanged;
their SHA256 hashes are pinned in
`third_party/benchmarks/volleybots/fixed-rally-protocol.json` and verified before
simulator startup.

This is a **World-authored cooperative-return opponent**, not the missing
official trained 1v1 baseline. Isaac Sim 6.0.1 compatibility is experimental;
official physics equivalence is not claimed. Native physics parameters, task
rules, and winner checks are not replaced. The original default adapter still
requires its official opponent manifest; the single-juggle profile is unchanged.

## Protocol

- Route: `T05` / `drone_volleyball_1v1`, profile `isaac6-scripted-1v1`.
- Default: model player1 receives from scripted player0; seed7; random_turn=false.
- Budget: 1000 native control ticks, 0.02 seconds per tick, no uncounted warmup.
- Score: native terminal `actor_1_wins`; loss/draw are zero, unfinished/infra
  runs are not native losses. Player0 runs are a separate protocol, not pooled.
- Model sees only its own symmetric 37D observation. Review video and opponent
  actions are logged for audit, not sent as policy observations.
- `CODE_CONTROL=off` by default in the dedicated script; `on` is a separate track.
  Offline Shell calculations remain available under existing isolation rules.

## Launch

From the repository root, with the existing isolated agent/GPU configuration:

```bash
MODE=zero STEPS=4 bash scripts/run_volleybots_1v1.sh --dry-run
MODEL=gpt-6-astra CODEX_AUTH_HOME=/path/to/auth bash scripts/run_volleybots_1v1.sh
```

`MODE=zero` or `probe` is diagnostic only, not a model score. Probe holds its
initial vector; it is not a closed-loop reference-policy match. This minimal
port does not add the remote shared `reference` execution mode or change the
campaign-pinned result aggregator. The generic launcher retains its existing
coding-control default; use this script or explicitly pass
`--disable-coding-control` for the off track.

Full task-specific fields remain in `result.json`: protocol, controlled player,
opponent hashes, outcome, native metrics and faults. Also retain `opponent.json`,
`native-config.yaml`, `both-player-actions.jsonl`, `compatibility.json`, event
logs and `video/review.mp4`. Generic summaries may omit task-specific fields;
use the raw result for this variant's analysis.

## Validation

```bash
python3 -m pytest -q environment/tests/test_volleybots_scripted.py \
  environment/tests/test_volleybots_wrenches.py \
  environment/tests/test_native_project.py environment/tests/test_rollouts.py
```

Offline tests cover immutable source hashes, default player/prompt permissions,
single/two-articulation force routing, action and win-side mapping, invalid
opponent outputs, draw/unfinished results and the existing native interfaces.
Local short GPU and source-Codex smoke acceptance passed on 2026-10-04:

- Probe: 8 ticks, 9 review frames, independently changing opponent actions,
  exit 0 and infrastructure_ok=true.
- Astra/Azure xhigh, coding off: 32 ticks via 7 apply_action calls, all 7 API
  responses completed, median full response 13.71 seconds, 33 review frames,
  no outgoing policy images, exit 0 and infrastructure_ok=true.
- Both runs stopped at the requested short budget with outcome=unfinished and
  success=null. They are diagnostics, not scored complete matches.
- Fresh private-cache startup spent several minutes compiling RTX shaders.
  Missing viewport.pxr registry messages did not prevent either run completing.

Artifacts are under `/root/robotworld-environment-setup-20260930/` in
`volleybots-scripted-smoke-1004-01` and `volleybots-astra-smoke-1004-01`.
Full-horizon match, native terminal outcome, and sustained opponent rally
acceptance remain pending. This change does not enqueue 1v1, restart running
episodes or modify existing campaign scores.
