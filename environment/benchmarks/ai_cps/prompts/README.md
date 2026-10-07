# Task-specific system instructions

`policy.Agent.run()` appends `catch.md` to the shared system prompt for task ID22. It describes the scene, observations, actions, and evaluation from pinned upstream source. It does not change simulation or scoring, or provide a prewritten controller.

Sources include `Franka_Ball_Catching.py` (initial conditions, observations, actions), `Models/Franka/Franka.py` (base rotation), `Models/ball_catching/tool.py` (tool-root initial position), `FrankaBallCatching.yaml` (frequency, gravity, velocity scaling), and the native monitor/optimiser.

This version was added after `four-cases-300steps-01`. A seed-7, 300-step rerun was recorded at `var/runs/docker/ai_cps/catch-prompt-v2-300steps-01`; it failed native scoring because the ball was not retained in the catching tool. Preserve each historical run's `prompt.json` rather than replacing it with this file. The shared prompt also corrects native velocity scaling to 0.1 for ball tasks; peg scaling remains 0.02.
