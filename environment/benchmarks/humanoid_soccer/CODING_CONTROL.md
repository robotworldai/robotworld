# coding_control：由模型编写的闭环控制片段

适用于需要快速反馈的局部动作：保持姿态、重心转移、落脚、分阶段踢球。LLM 负责写/修改控制逻辑；程序以 **50 Hz 仿真控制频率**接收新观测并输出一个关节动作。原版 PD 和可选踝关节辅助以 500 Hz 运行。代码执行期间不逐帧请求 LLM。

这是受限 Python 子集，不是任意 Python 或隐藏的行走技能。代码不能获得 simulator handle，不能读写环境文件、联网、重置、瞬移或修改物理。原 benchmark 与 Codex 源码均不修改。

## Tool schema

真实工具定义在 `control.py::coding_spec`；两种模式都提供 `coding_control`，与 `move_joints` / `review_action` 并存。

```json
{
  "type": "function",
  "name": "coding_control",
  "inputSchema": {
    "type": "object",
    "additionalProperties": false,
    "required": ["note", "max_steps", "code"],
    "properties": {
      "note": {"type": "string"},
      "max_steps": {"type": "integer", "minimum": 1, "maximum": 500},
      "code": {"type": "string", "minLength": 1, "maxLength": 16000}
    }
  }
}
```

上面省略了较长的 description 字段，实际描述会按 direct/hybrid 模式给出对应返回类型。

- `note`：本段目标及使用的反馈。
- `max_steps`：最多执行多少个 0.02 秒动作，共用 episode 预算，最多 500 步/10 秒；剩余 episode 不足时自动缩短。
- `code`：定义 `control(obs, memory)`。每个控制步调用一次，可定义辅助函数和常量，可使用 math。

## 输入与输出

`obs` 包含当前关节角/速度、骨盆姿态/高度/角速度、COM、脚踝位置和足地接触、球/球门相对向量、辅助修正及初始/上次名义目标。另有 `dt`、`program_step`、`program_time_s`、程序/episode 剩余步数、`joint_limits_rad`。Hybrid 额外包含**本步新预测**的原策略 action 与具名关节目标。没有图像或隐藏计分信息进入程序。

`memory` 是每次工具调用独立的可变字典，跨控制步保留。用它保存阶段、滤波量、名义目标与计数。下一次调用重新开始；需要延续时由模型显式带入代码。程序结果与 memory 必须可 JSON 序列化。

DIRECT 返回：

```python
return {"joint_positions": {"left_knee_joint": 0.2}}
```

这是绝对 rad，所有目标同时执行；省略关节沿用本程序上一步的目标，程序首步以实测值初始化。辅助可能继续修正踝关节。**不能反复把带辅助的实测踝角当成新的名义参考，否则可能累积修正。**

HYBRID 返回：

```python
return {"decision": "modify", "joint_offsets": {"right_hip_yaw_joint": 0.03}}
# 或 return {"decision": "accept", "joint_offsets": {}}
```

沿用原来的 ±0.25 rad 修正上限和关节限位，每步基于该步的原策略 action。

两种模式均可返回 `{"done": True}`：本步不执行动作，立即把控制权和新观测交回 LLM；不代表任务完成。超预算、错误、非法动作都会结束程序；已经执行的动作不回滚。程序结束只暂停仿真，不等价于真实硬件的急停制动。

## 推荐的 system prompt 表述

完整实际指令在 `policy.py::CODING_INSTRUCTIONS`，每轮保存到 `agent/prompt.json`。核心表达是：

> 对需要连续反馈的局部任务，优先考虑编写 `coding_control`。根据每个控制步的新姿态、接触和速度观测修正目标，而不是只根据初始观测生成整段动作。先用短时、小幅、有限速和退出条件的程序验证，再逐步扩大动作。插值保证目标平滑，不保证机器人平衡；平衡反馈也不保证单脚支撑成功。你需要设计阶段转换、动作和停止条件，并根据结果修订代码。

例如先进行短时站立观测（这只是保持名义姿态，不是步行控制器）：

```python
def control(obs, memory):
    if obs["pelvis_height_m"] < 0.55:
        return {"done": True}
    if "nominal" not in memory:
        memory["nominal"] = obs["initial_joint_reference_rad"].copy()
    memory["ticks"] = memory.get("ticks", 0) + 1
    return {"joint_positions": memory["nominal"].copy()}
```

## 执行边界与记录

代码通过 AST 解释器运行，不调用 Python eval/exec。支持 if/for/while、函数、字典/列表、推导式、有限 math 和数值内建函数；不支持 numpy、任意 import、类、生成器、lambda、f-string。完整可用集合见实际 prompt。

每回调最多 100000 个解释操作、2s 墙钟时间；单集合最多 4096 项，结果+memory 最多 64 KiB；独立 worker 限制地址空间和 CPU。它是仿真时间闭环，仍需测量时延才能评估真实硬件实时性。

- `agent/programs/program-XXX.py/.json`：原始代码和调用参数，包括执行失败的代码。
- `agent/events/environment.jsonl`：program_started/program_tick/program_finished、各步动作和物理反馈；包含程序返回的 memory。
- `agent/events/tools.jsonl`：工具请求与结果、已执行步数、退出原因。
- `agent/events/no-images/`：自动生成的无图版本。
- 视频仍按每个控制步录制，不随 LLM 调用稀疏采样。

测试同时覆盖提前 done、非法动作零步退出，以及剩余 episode 预算截断。代码工具只是新的动作生成入口，任务成功仍由原 benchmark 判据给出。
