# Bench2Dex adapter

RobotWorld-owned integration code is maintained here without modifying the upstream checkout.

- `project.py`: native scene initialisation, active-joint mapping, 60 Hz metric updates, 20 Hz tool stepping, RGB observations, and video.
- `compat.py`: Isaac 6 namespaces and rendering compatibility; retains tasks, success criteria, and actuator parameters.
- `assets.py`: pinned official Hugging Face downloads and hash verification.
- `anchors.py`: official first-origin scene anchors per task; metadata is read only on the environment side.
- `suite.py`: tasks 41–49, build/check/probe commands, and source-built runtime launch.

The tool and model loop reuse `../native_project/`. The model host and Docker simulator communicate through the isolated bridge. Evaluator ground truth is not sent to the model.

See the [complete Bench2Dex guide](../../../third_party/benchmarks/bench2dex/README.md).
