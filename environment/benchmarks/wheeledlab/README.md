# WheeledLab agent adapter

- `catalog.py` defines native configurations, RobotWorld courses, and their control budgets.
- `control.py` provides the tool schema and validation for native two-dimensional actions.
- `simulator.py` unpacks native policy observations, calls `env.step`, and records evaluation separately.
- `policy.py` supplies task instructions and observation history to the locally built Codex runtime, saving complete and image-free events.
- `camera.py` records review-only views at each control step; these views are not exposed to the policy.
- `custom.py` defines permitted onboard observations, task instructions, dynamic gates, and the evaluation bridge. Evaluator map, progress, and gate state remain hidden. Course cameras, terrain, and success criteria are implemented under `third_party/benchmarks/wheeledlab/robotworld/`, leaving upstream checkouts unchanged.

See [the integration README](../../../third_party/benchmarks/wheeledlab/README.md) for environment sources, containers, assets, and evaluation protocols.

## Episode records

Each episode saves `prompt.json`, `command.json`, `configuration.json`, `result.json`, original and image-free events, permitted observations, control programs, and `video/camera.mp4`. `native-observations/` stores per-step native policy vectors as NPZ files; visual tasks also save native greyscale PNGs. These review artefacts are not mounted into the agent workspace.

For custom courses, `capture.py` updates front, rear, and observer camera poses after each completed control step, then renders all products in one Replicator update. Each video contains one frame per control step. `camera-capture.jsonl` records step indices, frame counts, and physical time before and after rendering; advancing physics during rendering raises an error.

Camera intrinsics, viewpoints, physics frequency, and task criteria are retained. Native courses continue to use their original capture entry points. Each episode saves the interface and protocol source used for that run.
