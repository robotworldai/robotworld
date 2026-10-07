# omniisaacgymenvs adapter

T14: the original AnymalTerrain task and push disturbances in its separate legacy Isaac runtime.

- `project.py` loads the pinned environment, defines action documentation and permitted model observations, and reads native scores.
- [Source, assets, licences, and Docker recipes](../../../third_party/benchmarks/omniisaacgymenvs/README.md) are maintained separately from the adapter. Upstream source files are left unchanged.
- Trajectories, videos, and compatibility outputs are written under `var/runs/`, outside the source checkout.

Use `scripts/eval/omniisaacgymenvs.sh` for lower-level diagnostics, or consult the [integration overview](../../../docs/native17/README.md). The shared `../native_project/` loop provides `observe`, `apply_action`, and optional `coding_control`. Action dimensions, scaling, and observation semantics remain specific to the original environment.

Model runs use the locally built `codex/` runtime and Docker simulator. An implemented adapter does not establish environment readiness: consult the [validation record](../../../docs/native17/VALIDATION.md) for asset and hardware limitations and recorded episodes.

## Experimental Sim 6 runtime

`project_isaac6.py` is an optional experimental runtime; it does not replace `project.py`. It inherits the original actions, observations, scoring, and reset method, adding Sim 6 startup, namespace compatibility, and a review camera.

The review camera and light are persistent non-physical prims. A DomeLight is enabled only during zero-time `_capture`; `finally` hides it and resets its intensity. Assertions check visibility and the unchanged physics clock. The light remains hidden during action execution. The original `add_distant_light=False`, 188-dimensional state, and empty `policy_images` are preserved. Review video is not given to the model. The legacy gym extension is extracted from a locally licensed 4.0 image, with hashes and licence information retained separately.
