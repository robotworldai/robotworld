# BEHAVIOR-1K Original Source Entry

The actual ** environment code is in [checkout/](checkout/), not only this description file. ** Locally integrated complete version management source code: 4, 078 files, v3.9.3, commit `6cbf70b075816096e9be53958780769f3264d25d`. This is an independent work tree, not a soft link to World; 2026-09-26 check that `git status --porcelain` is empty.

## What's in the directory?

| Entry | Contents |
|---|---|
| [checkout/OmniGibson/](checkout/OmniGibson/) | Original simulation environment, robotics, scenes, sensors, controllers |
| [Official evaluator](checkout/OmniGibson/omnigibson/eval/evaluator.py) | Official episode Implementation and Evaluation Achieved |
| [Official R1Pro Configuration](checkout/OmniGibson/omnigibson/eval/r1pro.yaml) | Original robot and controller configuration |
| [checkout/bddl3/](checkout/bddl3/) | Definition of tasks, terms and objectives |
| [checkout/joylo/](checkout/joylo/) | Machine-related codes provided upstream with warehouses |
| [checkout/asset_pipeline/](checkout/asset_pipeline/) | Upstream asset disposal tool, non-asset capital |
| [checkout/docker/](checkout/docker/) | Upstream Docker Configuration |
| [checkout/docs/](checkout/docs/) | Upstream documents and challenge Job description |
| [checkout/README.md](checkout/README.md) | Upstream project description |

World's [adapter/tools](../../../environment/benchmarks/behavior_1k/), [Docker configuration](../../../environment/containers/behavior_1k/), and [evaluation bridge](../../../environment/integrations/behavior_eval.py) live outside the checkout. Authorized assets at `World/var/datasets/behavior_1k/`; Run record in `World/var/runs/docker/behavior_1k/`.

## Original environmental baseline

The layout, object, initial state, mission objective, physical parameters and rendering settings are based on a fixed version of the official environment and the official case asset. (b) Lack of reliance on retrofitting assets; The failure shall not be circumvented by deleting the object, changing the layout, changing the rendering or simplifying the task. The external Isaac compatible code is maintained separately and is not claimed as equivalent to the original environment.

The previous RayTracedLighting + customized IK short turn is only a historical connectivity experiment, and ** does not mean that the environment runs through **. A separate scene detection has been made through the original renderinger+official robotic configuration, and the reasons for the dark light remain to be identified, as detailed in [Status and borders](../../../environment/benchmarks/behavior_1k/STATUS.md).

## Source and Storage

There are already BEHAVIOR-1K warehouses local clone from the machine, which are not redownloaded or copied untraceed assets, caches, output. Git shares hard links with non-variable objects, and the source work tree is independent; alternates does not depend on the original directory. Fixed source recorded at [sources.local.json](../../sources.local.json). Do not edit checkout files or inject World descriptions or compatible patches into them.

The current checkout is ignored by the parent directory Git, which simply means that a source code must be arranged for release and does not mean that there is no local code. `World/upload-github/` is the old release snapshot, not the current development directory. This time, there is no transmission and no copy of another source code or asset.
