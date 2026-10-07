# BEHAVIOR scene

`carrying_in_groceries_dev.json`: Carrying groceries in house_double_floor_lower, completing the tailing of refrigerators and car trunks, using public_test index 10 (examples 311) for development inspection. The 5000 step development round (Q=0) to keep the original version of RealTimePathTracing was completed, and was unsuccessful. History RayTracedLighting is no longer the current configuration.

`clean_up_your_desk_dev.json`: The children ' s room desk is organized and items are returned, using house_single_floor, public_test Index 10 (examples 311). The latest desk-02 has completed the 5000 step budget round in accordance with the continuous trial rule, with the actual 5001 active control steps, 254 effective moves, no earlier waiver, official success=false/ Q=0.090909. The desk-01 advance give_up results remain historical. Use isolated source code Codex and Docker emulators to keep the scene and render. The asset readiness portal supports `--task-name clean_up_your_desk --instance-id 311`, the list is stored on `var/datasets/behavior_1k/subsets/` and assets are reused.

`turning_on_radio_dev.json` retains only other examples of scenarios for which no assets or measurements are downloaded.

Official reports use index 0 — 9, each time, to keep all results. The default timeout is determined by the official evaluator according to the length of the human presentation; Do not inherit RoboDojo step 1000. JSON is the scene description and the performance parameters are accepted by Docker launcher. An action and complete event is written to the rollout directory, and a copy of the `events/no-images/` analysis is automatically generated.
