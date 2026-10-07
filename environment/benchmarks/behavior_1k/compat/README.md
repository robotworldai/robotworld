# Isaac Sim 6.0.1 External Compatibility Layer

Here is the reviewed upstream PR #2313 difference, Kit configuration and visible import adapter. manifest.json fixed PR commit and original/target file SHA256; UPSTREAM_LICENSE retains upstream licences. Patch memory compiled, only adapted to the registration module, and third-party source files do not leave a disk.

Use `run.py --isaac601-compat --image world/behavior:isaac6.0.1-experimental` to enable; Only Isaac Sim 6.0.1.0 is accepted and the source file does not match. Each time compatibility.json runs, marked as an experiment. The default 5.1 entry is not enabled.

Simulator extra point Kit configuration sources to this directory; The rest of the `__file__` remains the original path, retaining material, metadata and resources relative to search. The source line number in the run-life log may be inconsistent with the original document and should be checked in conjunction with the fixed patch.

See ./ STATUS.md for status; Generating compatibility layers does not mean that the actual scene is passed.

The original environment baseline allows only `--renderer upstream` (default value). Historical RayTracedLighting replacement branch removed; The start-up parameter is no longer supported by the old command and the old running data only as historical evidence.

`--render-diagnostics` writes the original render call start, returns and time-consuming `render-events.jsonl` without changing the rendering settings. `--probe-only` uses the official R1Pro configuration and saves `render-state.json` (original exposure, lighting, and camera settings), plus 60-frame videos for each of three views and their first/last PNGs under `render-probe/`; This 60 frame does not advance physics, not mission trajectory. Model rounds are still configured using external IK tools and cannot be confused with official controller detection.
