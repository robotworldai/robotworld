Before acting, read `/workspace/ROBOTWORLD.md` for the full public task and robot specification, or call the `describe` tool. This file is generated from the live environment and is identical across harnesses.

Complete the wheel_legged wheel_legged_upright_recovery task. Read `describe` for the exact scene instruction, action definitions, units, reference frames, and native controller assistance. Use `observe` for the allowed images and proprioception. Use the exposed native robot tools with their exact schemas; they return observations after actions. If this run instead exposes `execute`, send ordered commands through that tool and query `observe` afterwards. Use `status` and `receipt` to inspect execution.

The configured world-state-v1 budget is 2000 native control steps. A command's `steps` repeats its native control input only where describe explicitly defines repetition. Some benchmark commands are planned macros, whose physical cost is measured from execution feedback. An action chunk can contain multiple commands. Execution stops on native success, native termination, or the step budget.

Continue until the episode is terminal. Textual claims cannot complete a task. No reset, privileged state, simulator filesystem, direct simulator handle is exposed. Feedback code is available only when describe lists coding_control. Only the configured simulator-side success checker determines reward.

## Robot and control reference


The public runtime describe response supplies the selected robot/controller profile, native action schemas and task-specific instruction. The RobotWorld Codex and Claude Code agents attach that response before the first model request. Other agents must read describe before acting. Do not guess missing joint order, dimensions, gains or coordinate frames. Use observe for measured state and images. Static examples describe requested targets, not guaranteed motion or a reference solution. Only tools in the runtime schema are enabled.
