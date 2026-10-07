# RobotWorld WheeledLab Custom scene v1 / Carborne Observation v2

The target difficulty layer is the 1 base capacity, 3 task. ** is not an official WheeledLab collection and does not pre-assure GPT to pass 1 and fail 3. ** determination fixed before model assessment; The pass rate needs to be measured. Keeps the original task, original vehicle USD and 2 D action source and does not modify checkout/Codex. This catalogue contains new environments, maps, terrain generation, rating and recosting tools.

2000 control step, 50 Hz (40 simulation), 200 Hz physics. Tools are still observe, drive, coding_control. LLM pauses simulations during reflection, so it is not a real-time reasoning test of the model 50 Hz wall clock.

** The current default observation has been changed to `robotworld-onboard-v2`: RGB + wheel speed/ turn encoder + IMU angle velocity and roll/ upside down. ** no longer provides global coordinates/yaw, real smooth/side slide speed, next checkpoint, check point progress, road centre line/complete map, barrier coordinates, friction partitions, gate position and pass conclusions. Encoder/IMU is still an ideal simulator and has not increased its noise and cannot be claimed to be fully consistent with real vehicle measurements. No GPS, laser radar or depth sensors.

At tool boundaries, the model receives a 640×360 front RGB image; At each control callback, the program receives 48×27 RGB8 pixels from that same camera as a flattened array indexed by `(y*48+x)*3+channel`, without extra semantic labels or lane detections. Tasks are given only to natural language destinations/rules and vehicle parameters. Car-borne video `video/onboard.mp4` and flashback third perspective `video/camera.mp4` are recorded with each control step; Only the former are strategic inputs. Historical observations are still 4 frames, interval2 (based on tool observations).

Old `custom-v1/*-gpt6-01` uses accurate state + public maps, first three successful; This is a comparison of the old conditions and cannot be written as a new version of a vehicle-borne visual accomplishment. See [Observation of boundaries and validation](ONBOARD.md) for the new version of the link.

New [Quite a 2-style driver.](PRECISION.md): Double beam bridge, narrow tunnel back into the garage, front- and back-to-side parking. Real-line contact is a failure, with an error of 2.5 cm in repository/side endpoint towards 3 °. Enables a real back and back-of-the-car lens using a separate precision protocol; Entry `bash scripts/eval/wheeledlab_precision.sh run ...`.

| ID | Target difficulty | Real sites and movements | Success conditions |
|---|---|---|---|
| rw-courtyard | Basic capacity | 2.6 m route, 0.15 m slope, 4 WD | Sequenced check points; Arrival of 0.65 m towards 25°, speed 0.12 m/s on 0.8 s; No failure |
| rw-hairpins | Difficulties | A 1.35 m-wide elevated road rising from 0 to 1 m, with two 180° hairpins of radii 2 m and 1.6 m | (a) Sequencingly passing checkpoints with high requirements; Parking distance 0.40 m, direction 12 °, low speed 0.8 s; No failure |
| rw-gate-dock | Difficulties | 6 s cycle real sports gates, speed-limit narrows, consecutive turns, loading boxes | 0.12 m/s continuous 0.5 s; 0.5 m passes when the gate passes. 0.65 m/s; Sequenced check points; Endpoint 0.32 m, heading towards 10°, keeping 1 s at low speed; No failure |
| rw-drift-switch | Difficulties | RWD closed track, four physical friction partitions, 0.6 – 1.1 s interval real speed/ yaw disturbance, inner container | The same circle passes all the checkpoints in sequence and crosses the finish line; Each turn requires at least 0.35 s of continuous valid sideslip: speed ≥0.7 m/s, body-forward speed ≥0.5 m/s, and sideslip angle 0.25–0.70 rad; No failures. One lap that does not meet the slide requirements can continue until the budget ends. |

Joint failure: body security envelopes crossing any corner of the road corridor, block intrusion barriers/gates, falling above 0.20 m below road altitude, facing/rolling > 60 °, orbital discontinuity, time lapse. The gates are also unchartered, wired when closed, and speed limit. The safety envelope extends ±0.30 m longitudinally and ±0.18 m laterally from the vehicle centre, with an additional 0.015 m obstacle margin. **, which is a clear geometric safety rule, does not pretend to record exposure. ** road/slop/barrier/gate itself has a PhysX collision; Building background outside the road. The risk of retrenchment of the original USD tyre in the cone of Isaac6 is still applicable and can be seen at higher level VALIDATION.md.

`specs.py` is the full numerical protocol and map source within the scorer, which is no longer sequenced to the model; `geometry.py` Generates real USD grids/crash/ rubbing materials; `environment.py` creates independent IsaacLab configuration; `scoring.py` does not rely on Isaac or model, but only determines success/failance based on the state of record. `onboard.py` provides a white list of vehicle-mounted cameras and sensors, and `sensor_markings.py` supplements visible red parking frames (no collisions/scoring) only to the gate. WheeledLab Docker does not require additional downloading of assets.

## Run

In World root directory:

```bash
bash scripts/eval/wheeledlab_custom.sh list
bash scripts/eval/wheeledlab_custom.sh run --model gpt-6-astra --codex-home var/auth/robodojo-codex
# Individual/Other models:
bash scripts/eval/wheeledlab.sh run --cases rw-hairpins --model YOUR_MODEL --codex-home /path/to/model-home
```

(a) The default set of original wheeledlab.sh is unchanged; These four are selected by special scripts or -- cases visible selection. - steps can only shorten the budget; incomplete does not count as success or official 2000 step assessment.

## Complete the evidence.

Each run saves scenario.json (evaluator map and thresholds), observation-profile.json, generated_assets/robotworld-course.usda, protocol-source/ and interface-source/ with SHA-256 hashes, events/scoring.jsonl (evaluator state/actions/checks), events/no-images/, result.json, two videos, and video/terrain-overview.png. The Codex tool and feedback are kept in events/tools.jsonl and the model process is in events/codex.jsonl. Save sensor-frames/front-lowres/ for the low-resolution original camera frame, complete events preserves the echo pixels, and version no-images only retains the image size/step/Hashi. Private map and score file is not mounted on agent sandbox.

```bash
python third_party/benchmarks/wheeledlab/robotworld/replay_score.py PATH_TO_RUN --check
```

(b) Recalculate the original read-only status using the decision code of the freeze round; Do not trust the model 's completed text, do not read the saved score to generate the answer, which is then compared to result.json. See VALIDATION.md for model results and solvency validation. These custom success fields are not available in the upstream environment and cannot be confused with official success rates.
