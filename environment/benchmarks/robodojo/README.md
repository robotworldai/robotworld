# RoboDojo adapter

## Shared Visual Memory

New runs share `current-plus-4-history-stride-2-v1` with BEHAVIOR: current
observation plus up to four prior rounds at interval two. The request boundary
keeps only the latest packet's images and preserves earlier text/tool records.
All cameras are retained per selected round; three cameras mean at most fifteen
images per request. Full images/video stay on disk. `request-audit.json` records
the selected rounds, incoming/outgoing counts and hashes. Failed image auditing
rejects the run rather than producing a valid model score.

For GPT-6 Astra, the existing private AI Hub credential is used by the per-run
direct route, bypassing the legacy shared image-dropping proxy. Other models
retain their configured provider. No global Codex authentication is changed.
The new source path requires runtime verification before claiming parity with
the historical container results below; previously built images are not rebuilt
or silently updated by this source change.

move_eef with source code Codex has driven the real RoboDojo / Isaac Sim 6.0.1 inside Docker to complete hand lift, return, trap and cross-border refusal. Use the user-authorized [External Compatibility Layer](compat/README.md) without modifying the upstream source code; The most recent transfer belt matching capture has been officially successful, and continuous video and prompt against evidence has been preserved. See [Authentication status](STATUS.md) for details.

- [Run Mode](RUNNING.md)
- [Actual Authentication Status](STATUS.md)
- [Docker Build Package and Run Entry](../../containers/robodojo/README.md): Independent local snapshots constructed and measured.
- adapter.py: Open TASK_ENV observation, planning and implementation interface.
- eef_executor.py: Pure EEF implementer extracted from RoboProbe, no LLM client.
- deploy.py: Directly available eval_one_episode.
- docs.py, types.py, spec.py: robotic description and data contract.
- SOURCES.json, LICENSE.RoboProbe: Extract source and license.

The Codex runtime is in ../../runtime and the external entry point is ../../integrations/robodojo_smoke.py. Do not modify the upstream source code, and do not add files to XPolicyLab.policy.
