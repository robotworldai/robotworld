# 上电站立与策略接管测试

## 目标与边界

这个功能处理的是机器人以可恢复的低姿态放在地面后，控制器如何安全上电并进入
正常运动策略。它不是完全侧躺自起立策略，也不替代 Recovery：

- Power-on Stand 负责无力矩到正常力矩的确定性启动过程；
- Recovery 负责已经上电时的小到中等失衡、推撞和接触扰动；
- 完全侧躺、趴倒或翻转后的 Self-righting 仍需单独做机械可行性与策略训练。

状态机使用普通 6 维 VMC 动作，不引入新的 PPO 环境：

```text
PASSIVE → RAMP → EXTEND → STABILIZE → HANDOFF → COMPLETE
  0力矩    力矩渐入   平滑伸腿      保持稳定      策略混合接管
```

VMC 动作项新增逐环境 `motor_enable_scale`。其默认值为 1，因此已有 Train、Play
和 checkpoint 行为不变；只有测试或未来实机启动管理器主动调用接口时才会关断或
缩放输出。力矩为零时轮速 PI 积分也停止并清零，避免上电瞬间积分饱和。

## 快速运行

以下命令从项目根目录执行：

```bash
export ISAACLAB_ROOT="/absolute/path/to/IsaacLab"

"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/power_on_stand_open_loop_test.py \
  --headless
```

默认并行测试 72 个工况：

- 无动力等待：0.4、0.7 s；
- 初始基座高度：0.20、0.25 m；
- roll：-5°、0°、5°；
- pitch：-8°、0°、8°；
- 左腿关节不对称偏移：0、0.12 rad。

先做单工况冒烟测试：

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/power_on_stand_open_loop_test.py \
  --passive_times 0.5 \
  --initial_heights 0.22 \
  --initial_roll_degs 0 \
  --initial_pitch_degs 0 \
  --left_leg_joint_offsets 0 \
  --headless
```

参数可使用逗号给出多个值，例如：

```text
--initial_pitch_degs=-10,0,10 --passive_times=0.3,0.6,1.0
```

负数列表建议使用 `--参数=值` 的形式，避免 argparse 把负号误识别为新参数。

## Play 中运行时断电重启

使用任意支持键盘 Play 的 VMC 环境运行已有策略：

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Wheel-Legged-Recovery-Flat-v0 \
  --checkpoint /absolute/path/to/model.pt \
  --keyboard \
  --real-time
```

进入 Viewport 后：

- 第一次按 `K`：清零移动和跳跃命令、清空控制器状态，并将电机输出置零；
- 第二次按 `K`：开始安全上电状态机；
- 出现 `Policy handoff complete` 后，恢复正常键盘运动；
- 上电过程中再次按 `K` 会立即重新断电。

重新上电前必须检测到两轮接地且倾角不超过 0.60 rad；否则保持断电并提示等待。
可以用 `--power_restart_max_tilt` 调整倾角门槛，但不建议为了让完全侧躺状态强行
通过而放宽它——那属于尚未实现的 Self-righting 技能。

Play 参数 `--power_ramp_time`、`--power_extend_time`、
`--power_stabilize_time`、`--power_handoff_time` 和 `--power_target_length`
可以覆盖默认启动时序。该功能模拟执行器失能和控制状态重启；Isaac Sim、传感器
和 Python 进程仍在运行，因此不等价于整机电池、计算机和通信总线全部断电。

## 与训练策略交接

不传 `--policy` 时，HANDOFF 阶段继续保持确定性站立参考，用来单独验证上电
物理过程。传入由 `scripts/rsl_rl/play.py` 导出的 TorchScript `policy.pt` 后，
控制器会用 smoothstep 在 HANDOFF 时间内从站立动作连续混合到策略动作：

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/power_on_stand_open_loop_test.py \
  --policy logs/rsl_rl/<experiment>/<run>/exported/policy.pt \
  --headless
```

应优先使用 `Wheel-Legged-Recovery-Flat-v0` 或与测试环境观测维度一致的策略。
如果 JIT 策略只接收旧观测向量的前 N 维，可临时加
`--policy_observation_dims N`；这只适合明确知道观测排列没有变化的兼容场景，
不应靠截断掩盖任务或观测语义不一致。

## 默认时序

| 阶段 | 默认时长 | 行为 |
|---|---:|---|
| PASSIVE | 0.4/0.7 s | `motor_enable_scale=0`，让机器人自然沉降 |
| RAMP | 0.60 s | 平滑把力矩门控从 0 增至 1，保持沉降后的腿长 |
| EXTEND | 0.80 s | 平滑伸到 0.21 m 目标腿长 |
| STABILIZE | 0.60 s | 保持直立参考并等待动态衰减 |
| HANDOFF | 1.00 s | 混合到 JIT 策略；未提供策略时保持站立 |

可通过 `--torque_ramp_time`、`--extend_time`、`--stabilize_time`、
`--handoff_time`、`--post_handoff_time` 和 `--target_length` 调整。默认在混合
结束后继续评估 1.5 s，避免把策略接管瞬间的短暂调整误判成最终失败。实机首次
测试应使用比仿真更慢的渐入和伸腿过程，并同时施加硬件级电流、力矩、速度和
关节限位。

确定性站立参考还包含一个小型俯仰 PD：它根据 IMU 的 pitch 和 pitch rate 给
左右轮对应相同底盘方向的速度参考（URDF 关节轴镜像，因此关节符号相反），使轮
接触点移动到倾斜机身下方。默认参数是
`--balance_pitch_kp 3.0 --balance_pitch_kd 0.45`，输出受
`--balance_wheel_action_limit 0.65` 限制。该反馈仅用于安全启动和无策略开环
验收。伸腿过程中还会用 `--balance_leg_angle_gain -1.0` 尝试保持虚拟腿相对
世界竖直，避免基座仍接触地面时只沿倾斜机身方向顶起。两者都不替代学习策略中
的完整平衡控制。

## 输出与通过条件

结果默认保存在：

```text
power_on_stand_results/<timestamp>/
├── metadata.json
├── summary.csv
└── trace.csv
```

- `summary.csv`：每个初始工况的最终结论和最坏安全指标；
- `trace.csv`：按时间记录阶段、力矩门控、姿态、高度、腿长和接触；
- `metadata.json`：总成功率、平均首次稳定时间、状态机配置和判据。

默认要求至少出现连续稳定 0.5 s；最终通过还要求 HANDOFF 完成后的评估窗口中
至少 80% 控制帧稳定，且测试结束时仍满足：

- 倾角小于 0.20 rad；
- 基座高度处于 0.17–0.25 m；
- 稳定帧中两轮接触地面且基座不触地；
- 基座角速度模小于 0.50 rad/s。

这些是自动回归判据，不等于实机安全认证。还应重点检查
`max_base_contact_force_N`、`max_wheel_contact_force_N`、
`max_torque_saturation`、`minimum_joint_margin_rad` 和 `max_abs_vz_mps`。

当前代码在本机 seed 42 的默认 72 工况验证中，确定性站立保持通过
`72/72`；接入现有 Recovery JIT 策略后通过 `70/72`（97.2%）。后两例最终
姿态正常，但接管后稳定帧比例分别约 0.78 和 0.76，低于严格的 0.80 门槛。
这组结果只证明当前仿真链路可运行，不代表不同硬件参数、随机种子或实机也会有
相同成功率。

## 推荐验收顺序

1. 单工况、无策略、慢时序，确认力矩门控和腿长轨迹正确；
2. 扩大到默认工况网格，调整 ramp/extend 时长，不修改 PPO；
3. 加入 Recovery JIT 策略，检查 HANDOFF 是否产生动作或轮速突变；
4. 仿真中加入质量、摩擦、电机强度、延迟和传感器噪声扫描；
5. 实机悬空验证电机方向和限幅；
6. 使用吊架、急停和低电流限制做首次接地上电；
7. Power-on 与 Recovery 都通过后，再讨论独立 Self-righting 技能。

不要用仿真中的成功率替代机械限位、急停、支架和人员隔离。
