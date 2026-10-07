# AI-CPS adapter

This directory contains RobotWorld's policy and interface code:

- `simulator.py`: invokes the original tasks.
- `policy.py`: system instructions and the model interaction loop.
- `control.py`: native tools and the task ID34 criterion.
- `camera.py`: continuous video recording.
- `scoring.py` / `score_worker.py`: invoke the original STL monitor.

Environment source, assets, Docker recipes, and compatibility code are in `third_party/benchmarks/ai_cps/`.

Generated control reuses the restricted `humanoid_soccer.coding.ControllerProgram` interpreter without loading the soccer environment or policy. Franka action semantics are defined locally.

The lower-level entry point is `bash scripts/eval/ai_cps.sh list|sources|assets|build|run|all`. Its historical diagnostic suite includes four cases; use the top-level batch catalogue for the release evaluation selection. See the [project README](../../../third_party/benchmarks/ai_cps/README.md) and `TASKS.md` for installation, protocols, and limitations.
