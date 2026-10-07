# Codex → move_eef → RoboDojo

实现位于本目录与 `environment/runtime`。没有自写模型 API 客户端；模型推理由本地源码构建的 Codex 执行。

## 当前推荐：独立 Isaac Sim 6.0.1 镜像

镜像 `world/robodojo:isaac6.0.1-local` 已构建，使用已授权的[外部兼容层](compat/README.md)。

```bash
export WORLD_ROOT=/opt/robotworld/World
export WORLD_CODEX_HOME="$WORLD_ROOT/var/auth/robodojo-codex"
bash "$WORLD_ROOT/environment/containers/robodojo/run-conveyor.sh" smoke
bash "$WORLD_ROOT/environment/containers/robodojo/run-conveyor.sh" task
```

前者为四步工具检查，后者为官方传送带指令。命令使用镜像内的 Python / RoboDojo / IsaacLab，不挂载这些宿主依赖。完整说明见 [Docker README](../../containers/robodojo/README.md)。

## 构建与检查 Codex

在 World 下执行：

```bash
python environment/scripts/build_codex.py
```

构建脚本使用 `var/build/toolchain/cargo/bin/cargo` 与对应的隔离 RUSTUP_HOME，版本遵循 codex/codex-rs/rust-toolchain.toml；目标为 codex-app-server。首次需安装该工具链，产物记录为 `var/build/codex/<commit>/build.json`。不回退全局 codex，不改上游锁文件。

```bash
PYTHONPATH="$PWD" python -m environment.scripts.check_codex \
  --manifest var/build/codex/<commit>/build.json \
  --output var/runs/codex-protocol-check
```

此检查只注册工具，不发起模型推理。认证/provider 通过 Codex 正常的外部配置和运行时环境提供；桥接程序不复制或处理 API key。

若要单独验证真实模型的工具调用与图像返回（不连接仿真，不代表机器人测试通过）：

当前机器的模型目录将模型限制为 `code_mode_only`。本桥接使用 Codex 原生动态工具；使用公开配置生成独立的 `direct` 模式目录副本，保持模型/provider 不变，不覆盖用户原配置：

```bash
PYTHONPATH="$PWD" var/venv/bin/python -m environment.scripts.prepare_direct_catalog \
  --source /home/robotworld-user/.codex/models-gpt6-0.154.json \
  --output var/configs/models-direct.json --model gpt-6-astra
```

若副本已经存在可直接复用，脚本有意拒绝覆盖。当前源码可构建 app-server；可选 Code Mode helper 的 V8 下载返回 404，因此此处采用 direct 工具模式。

```bash
PYTHONPATH="$PWD" var/venv/bin/python -m environment.scripts.check_codex_tool \
  --manifest var/build/codex/<commit>/build.json \
  --output var/runs/codex-tool-protocol \
  --model-catalog var/configs/models-direct.json
```

这个诊断只让 Codex 调用一次校验 handler，并返回明确标记为合成的图像；不发送机器人 action。

## 已有 TASK_ENV

在自行 reset 的 RoboDojo 环境进程中，先补齐官方 `run_eval()` 的奖励/评分注册前置调用。如果已经由官方 `run_eval()` 调用，则不要重复注册。

```python
from environment.benchmarks.robodojo.deploy import eval_one_episode
from environment.benchmarks.robodojo.lifecycle import prepare_episode
from XPolicyLab.utils.process_data import decode_image_bit

prepare_episode(TASK_ENV)
result = eval_one_episode(
    TASK_ENV,
    manifest="/absolute/World/var/build/codex/<commit>/build.json",
    output_dir="/absolute/World/var/runs/robodojo/smoke",
    decode_image=decode_image_bit,
    max_actions=4,
    timeout_s=300,
    config_overrides=['model_catalog_json="/absolute/World/var/configs/models-direct.json"'],
)
```

调用者拥有环境 reset/close。这个低层入口未达到官方终止时 `official_success` 为 null，不能把模型 final 当作成功。上面的 task 启动器还会调用 `finish_task_episode()`，按 RoboProbe 原逻辑将 policy 提前停止的未完成任务结算为失败；smoke 不作该任务结算。

## 从外部启动官方环境

使用兼容的 RoboDojo Isaac Python，设置 PYTHONPATH 为 World、RoboDojo 和 RoboDojo/XPolicyLab 三个绝对路径；从任意目录运行：

```bash
python -m environment.integrations.robodojo_smoke \
  --root /absolute/RoboDojo \
  --output /absolute/World/var/runs/robodojo/move-eef-smoke \
  --manifest /absolute/World/var/build/codex/<commit>/build.json \
  --model-catalog /absolute/World/var/configs/models-direct.json \
  --task general_pickup --layout 0 --headless --enable_cameras
```

`--probe-only` 替代 manifest 可单独检查环境初始化和图像。正常 smoke 要求 Codex 依次抬左手 1 cm、回原高度、打开右夹爪、发送一次越界目标验证拒绝。最多四次动作请求；不可达目标不会移动。

环境通过 `create_eval_env` / `reset` 的公开接口启动，然后直接调用我们的入口。不往 XPolicyLab.policy 增加文件。官方 EvalEnv 强制连接的 WebSocket 仅提供 reset/handshake 占位，get_action 明确拒绝；全部动作来自 Codex 动态工具。

## 验证与产物

```bash
PYTHONPATH="$PWD" var/venv/bin/python -m pytest -q environment/tests
```

工具 trace 在 episode.json，RGB 帧在 frames/，Codex 进程错误在 codex.stderr.log。planner 参数错误与不可达反馈可交给 agent；基础设施异常会抛出并归档。基准源码和 Codex 源码文件均不修改；Isaac 6.0.1 的外部运行时兼容层已获用户授权。

单元/回归测试通过不代表真实 Codex 推理或仿真成功；实际状态以 STATUS.md 为准。

## 最近观测窗口

外部 adapter 默认每次返回当前三路 RGB，以及前 4 组历史图像、间隔 2 个观测轮次：`t-8, t-6, t-4, t-2, t`。开局不足时只返回已有帧，不填充假历史。每组标注 CURRENT/HISTORY、观测序号和实际环境动作步数；动作长度不等，因此该间隔不是固定仿真时间间隔。对应 `frames/<n>/history.json` 可审计实际选帧。

该窗口增加当前工具结果中的显式视觉上下文，不裁掉 Codex 会话中的旧消息，不能视作上下文成本上限。实现仅修改外部适配层。

## 连续录像与 prompt 对齐

正常 smoke/task 的 `video/cam_head.mp4`、`cam_left_wrist.mp4`、`cam_right_wrist.mp4` 逐个环境动作步采集，以实际控制频率（本场景 25 Hz）编码。重复 get_obs 不追加重复帧，LLM 推理等待不推进物理、不写入虚假的运动时间。`video/manifest.json` 记录每个环境步、实际帧数、帧率和编码结果。旧 `head-observation-replay.mp4` 仅为稀疏观测幻灯片，不是连续录像，不能据此判断仿真卡顿。

System prompt 现在逐字复用 RoboProbe `RoboDojo_Agent_L3_Inspect_EEF/policy.py::_system_message`，按本轮预算填值；move_eef 描述/schema 与该文件对照测试。2026-10-01 按用户要求，RoboDojo 不再提供父 policy 的可选 give_up 工具；本地代理和直接请求入口均只声明 move_eef，未声明的 give_up 调用不能结束 episode。14 维状态、关节上下文、到达误差、embodiment docs 延用提取的参考实现。历史帧和无 give_up 工具的说明另行追加；Codex developer instructions 保留环境工具限制。每次真实模型运行将实际 system/developer/tools 保存到 `prompt.json`，以便复核。Codex 的 agent 循环、认证和会话协议仍由源码 app-server 承担，没有复刻 RoboProbe 的 HTTP client；工具结果封装是 Codex dynamic tool 格式。

移除工具不改变原生成功、失败、步数或墙钟预算，也不自动启用文字终答后的续轮。此前带 give_up 的运行记录保留原样；新工具条件如需重测，应使用新的 batch，不能把历史结果改写为新条件结果。

只读环境诊断（无模型、不是任务评测）：

```bash
python3 environment/containers/robodojo/run_isaac601_local.py probe \
  --image world/robodojo:isaac6.0.1-local \
  --output var/runs/docker/isaac601-conveyor/my-direction-probe \
  --diagnostic-steps 100
```

机器人保持原位，记录物体实际位移与带面速度/坐标变换到 `scene-diagnostic.json`；这份诊断真值不送给模型。录像仍逐步采集。
