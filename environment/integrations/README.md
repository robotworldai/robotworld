# Integration entry points

This directory contains external policy packages, official plugin entry points, evaluator runners, and host/container RPC bridges. Launch integration belongs here; semantic adaptation belongs in `benchmarks/`.

Prefer upstream extension points. For RoboDojo's fixed `XPolicyLab.policy` import path, verify external loading without inserting files into the upstream package. Record any unsupported API boundary rather than hiding it in an unrecorded patch.

Container bridges should carry `episode_id`, `call_id`, and `observation_id`, and handle timeouts, cancellation, and duplicate requests. Invoke simulation APIs on the threads permitted by the environment.
