# Benchmark adapters

Each subdirectory maintains the relevant adapter, observations, executor, optional deployment entry point, documentation, and compatibility records. Shared interfaces are described in `../docs/contracts.md`; upstream implementations are stored in `third_party/`.

- [Bench2Dex](bench2dex/README.md) integrates RobotWorld tasks 41–49 using native bimanual/dexterous active-joint control and per-physics-step MetricTracker updates.
- [Native-project integrations](../../docs/native17/README.md) retain project-specific task construction and stepping. `native_project/` shares the source-built agent tool loop, action validation, and trace recording; it does not replace every environment's `step` implementation.
- [HumanoidSoccer](humanoid_soccer/README.md) supports 29-joint G1 policy review/correction and direct joint control.
- VolleyBots includes the independent official single-drone juggling task `T05-single`, launched through `scripts/eval/volleybots_single.sh`. The 1v1 task still requires its own complete opponent.

`operating_brief.py` contains shared public instructions. Action gains, coordinate frames, and scoring remain task-specific. See the [prompt and review audit](../../docs/PROMPT_AND_REVIEW_AUDIT.md).

RoboDojo's end-effector interface does not imply that other robots use CuRobo, two six-degree-of-freedom arms, or the same action space.
