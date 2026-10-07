# No LLM Full verification

Here's a list of World-owned scenes, semantic audits, positive and negative tests, and batch operators.

- `selected-tasks.json`: Four core benchmark missions selected by users in this round, as well as other project tasks that have been accessed; An independent record of historical aliases.
- Run data, continuous MP4 and gradual determination at `var/runs/validation/`; Summarized in `reports/validation/`.
- Upstream codes and success stories remain unchanged. The artificial state used to test the evidence is a diagnostic kit and does not count as a robot.
- Loading/step-through, word-backing, physical consistency verification are three independent states and cannot be substituted for one another.

## Batch Run

Execute in World root directory:

```bash
bash scripts/validate_all.sh --output var/runs/validation/my-check --dry-run
bash scripts/validate_all.sh --output var/runs/validation/my-check --bench robocasa
```

`--bench all` check all selected entry points; `--cases` can further select tasks (see `--help`). GPU environment string running without calling Codex or model API. The complete command, return code and output directory are written to `campaign.json`, and the remaining tasks are checked after a single job error.

The core four benchmark defaults for scene diagnosis using 300 control steps; The remaining entry points are subject to the registered original deadline. Diagnosis time limits are recorded separately from official rounds. Early termination of the reservation at birth does not remove the conditions of fall or crossing the border for the purpose of extending the video. Bench2Dex maintains the current joint position using the selected initial state and original round budget.

Simulation of leaf nodes for `predicates` scripts only validates the combination, threshold and direction of actual findings; The artificial simulation of `physical_fixtures` is recorded separately and cannot be counted as a robotic success. The video covers the entire diagnostic round actually carried out and the historical frame sample is not involved in video sampling.

Here, "no model call" means no GPT/Codex/API decision-making. Football uses upstream ONNX strategies as primary baselines; Script reference controls are used for the three tasks of precision driving, while the remainder are maintained/zero by the entrance. This diagnosis cannot be considered a success rate for GPT.


## Rating and video review

`report.py` summarizes the completed results; `video_audit.py` checks actual MP4 files with ffprobe; Run `frame_audit.py --report-dir reports/validation/Date` after refreshing the first two, checking the number of step-by-step video frames and returning non-zeros with missing or inconsistent returns.

- `ai_cps_score_replay.py`: In the AI-CPS dependent mirror, no GPU recalculates the original STL sequence and custom contact recovery count.
- `bench2dex_score_replay.py`: Restart the final and stage test with the state of each physical step object and do not recalculate the safety indicators that lack the full limit baseline.
- `driving_score_audit.py`: Freezing the rating source code to redisplay seven self-defined driving tracks, three sophisticated driving rounds and an independent polygon algorithm to check the declared security frame and line.
- `*_predicates.py` / `*_contracts.py`: Synthetic input test for the actual original function; It is clear that it does not pretend to be a simulation of physics.

Run failed without overwhelming the original log. Batch completion tags only indicate that all entry points have been tried and do not represent full success or physical equivalent.

Each registration entry point in this cycle only checks the scene/seeds fixed in the command. Operation complete does not represent all random layouts available; This should be expanded by creating new batches and keeping original running records.
