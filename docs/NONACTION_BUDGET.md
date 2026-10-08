# Non-action interaction budget (staged, not production-enabled)

## Campaign rollout (2026-10-02, pending launches only)

Following explicit user approval, the local campaign dispatch now selects
`nonaction-20-150-v1` for not-yet-launched Opus/K3 tasks and Astra reruns.
Existing occupied GPU episodes are unchanged. The shared local GPU launcher
forwards the setting into the policy container; `launch.json` records it.
The repository default remains off. This campaign uses the approximate
`observable-events-v2` counter described below, not exact built-in invocation
admission. Pending work and native physics receipts drain before interruption;
an internal built-in call may race the host interrupt. Full physical cleanup
acceptance across all backends remains outstanding. Inspect failures as
infrastructure errors rather than fabricating valid scores. Keep protocol
labels on new results and retain all legacy results separately.

## Observation mode (2026-10-02)

Set `WORLD_NONACTION_PROTOCOL=observe` in a **new policy process** to record
candidate counters without imposing an interaction limit. Default remains
`off`; no running episode, official queue configuration or historical score
has been changed. This mode does not inject budget instructions, interrupt
turns, add continuation prompts, change legacy end-of-answer handling, or
convert native results to interaction-budget failures.

`interaction-budget.json` records `mode`, `accounting_version`, totals,
`consecutive_peak`, `counts_by_kind`, and the first candidate threshold
crossing (turn ID, confirmed control steps and counters). `stop_reason`
remains null in observation mode, even after 20/150. These are candidate
thresholds, not calibrated limits.

Observable auxiliary starts/completions share one item key, so a failed
command or a command still running is counted without charging its later
completion again. Identical terminal-interaction notifications are counted
separately, since identical polls may be distinct calls. Zero-step dynamic
receipts and empty normal turn completions count; failed API turns do not
add an empty-turn charge. A tool-bearing turn is not charged again on exit.

This is deliberately approximate event telemetry, not exact invocation
accounting. `counts_complete=false`, explicit limitations, and
`missing_turn_end_count` expose known gaps. Missing invalid-call/final-poll
events may undercount; duplicated terminal notifications may overcount.
No Codex rebuild or strict tool admission gate is included. Compare Astra,
K3 and Opus under the same accounting before selecting enforced limits;
old Astra counts are not proof that 20/150 is compatible.

## Protocol

Candidate version: `nonaction-20-150-v1`. Control-step, task, checker and
wall-clock budgets remain independent. No model finish/give_up tool is added.
The intended accounting is one unit per auxiliary invocation (including
failures), one unit per robot invocation with zero confirmed control steps,
and one unit per normally completed turn without either progress or counted
calls. Text chunks and retry notifications do not count. Confirmed control
steps reset the consecutive count, never the total. Either consecutive >=20
or total >=150 latches termination. Already executing actions retain their
receipts; no extra hold, release, reset or scoring motion is performed.

The staged implementation is opt-in via
`WORLD_NONACTION_PROTOCOL=nonaction-20-150-v1` in the **policy process**.
Default is `off`; the official queue has not been changed to enable it.
Container launchers do not yet uniformly forward this option. Do not treat
setting it in an arbitrary host shell as proof of effective configuration.

## Implemented

- Shared counters and versioned `interaction-budget.json` per policy session.
- Zero-step dynamic tool receipts, auxiliary lifecycle notifications and
  empty completed turns, with observed duplicate-event suppression.
- RoboDojo continuation instead of final-answer termination when enabled.
  Its older 3-empty-turn/64-no-motion guards do not override this protocol.
- Hooks in RoboCasa, RoboLab, BEHAVIOR, native-project, AI-CPS, WheeledLab and
  HumanoidSoccer policy loops, plus explicit budget-stop result paths.
- Native success evidence is retained, and missing native full-window
  evidence is not fabricated. A protocol failure is separate from native
  checker completion. Historical files and scores are not rewritten.

## Required before enabling

1. Complete auxiliary invocation observability/admission. The pinned Codex
   notifications do not expose every invalid built-in call or final empty
   write_stdin poll. commandExecution completion may describe process exit,
   not the end of an individual exec invocation. terminalInteraction can
   lack a unique write_stdin invocation ID. Notification-only counts are
   therefore not a strict implementation of the intended contract.
2. Prevent newly generated built-in calls after the limit, while draining
   previously admitted work. A post-hoc application notification alone is
   not an execution admission gate, especially with parallel built-in calls.
3. Physically validate budget termination and cleanup for each evaluator
   family, especially RoboLab recording/summarization and BEHAVIOR metrics.
4. Thread an explicit protocol option through formal scripts, launch
   manifests and containers. Reject unsupported versions. Update campaign
   audit to distinguish legacy and new-protocol coverage without overwriting
   any previous result. Re-audit Astra including zero-step calls; its earlier
   auxiliary-only audit is not proof of compatibility.

## Verification so far

111 focused unit/integration tests passed on2026-10-02. Real pinned Codex
image-window-v2 with a synthetic provider completed exactly20 empty turns,
then stopped with nonaction_consecutive_limit and0 physics steps.
The evidence is retained in the local validation archive, outside this repository.
This is not all-backend physical acceptance. Broader tests also exposed
pre-existing mismatches for image-policy recovery and the AI-CPS catalog;
those policies were not changed by this work.
