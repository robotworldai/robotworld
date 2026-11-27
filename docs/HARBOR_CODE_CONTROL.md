# Harbor Shell and feedback control

Shell and file tools are enabled in both RobotWorld Harbor agents. They run in the agent container, for calculations, permitted observation processing and local files. They do not grant simulator filesystem access or permission to bypass the episode action API. Codex uses workspace-write; Claude Code enables Bash, Read, Write, Edit, Glob and Grep.

Feedback control is a separate, simulator-enforced setting:

```bash
# Default: direct action commands plus offline shell
export WORLD_CODE_CONTROL=0
# Optional: enable bounded feedback programs
export WORLD_CODE_CONTROL=1
harbor run --config configs/jobs/robotworld-codex.yaml
```

Use the corresponding Claude Code job to select that harness. Set the variable before launching Harbor; it is passed into each simulator container through Compose. Existing trials keep their original setting. For different settings per task, launch separate jobs with different environment values or explicitly set the simulator's WORLD_CODE_CONTROL to "0" or "1" in that task's Compose file. The setting is global to a launch only by convenience, not a model-selected permission.

`describe` lists `coding_control` only when enabled. Disabled calls are rejected by the native adapter. The simulator archives the setting in execution-config.json. Both harnesses receive the public runtime description before their first model request.

## Program contract: harbor-command-return-v1

Submit a normal execute command with tool `coding_control` and arguments `code`, `max_steps` (1–600), and `note`. Define `control(obs, memory)` in the restricted Python subset. Each callback receives the native public observe payload, excluding images, plus `program_step`. Numeric state fields remain bench-specific; inspect observe first. Memory persists within a program call. Return either:

```python
{"tool": "<native tool from describe>", "arguments": {"...": "..."}}
# Or end this segment without declaring success:
{"done": True}
```

Each command must prepare exactly one action with one repetition (`steps=1` where applicable). Native planner macros such as move_eef may still consume multiple physical steps: max_steps limits callbacks, not elapsed simulation time. The episode's native termination and control-step cap remain authoritative.

RoboCasa example (an interface example, not a solving controller): use the exact move_eef schema from describe to construct the returned arguments, with steps=1. Do not copy arguments from another benchmark. A portable no-motion callback is:

```python
def control(obs, memory):
    return {"done": True}
```

The worker supports bounded Python arithmetic, collections and math, without arbitrary imports, files, networking or simulator handles. Returned commands go through the same native prepare/step functions. Code or command errors stop the segment before that callback's physics step and appear in the receipt; previous actions cannot be undone. Native step errors remain infrastructure errors. Programs and callback observations/results are recorded in simulator events.jsonl, and videos use the normal simulator recorder.

This new service contract is not the old per-benchmark targets/action-return schema. Port legacy programs explicitly. It does not add sensors or automatically stabilise a robot.

## 中文说明

Shell 已开启；code control 由 `WORLD_CODE_CONTROL=0/1` 独立控制，默认关闭。开关需要在启动新 trial 前设置。程序只能读取公开观测、返回原生动作命令；不能重置、瞬移、修改评分或直接操作模拟器。每题可以在 Compose 中固定开关，也可按不同配置分 job 运行。此处 max_steps 是回调次数上限，规划器宏动作不保证一回调等于一个物理步。旧版返回 targets/action 的程序需改成 tool/arguments 格式。
