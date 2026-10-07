# Wheel-Legged-Jump-Flat-v0 任务设计与训练手册

## 1. 当前实现的定位

`Wheel-Legged-Jump-Flat-v0` 是独立于 `Wheel-Legged-Flat-v0` 的跳跃训练任务，但复用同一机器人、VMC 动作空间、平地观测和已经验证过的平衡能力。

第一版只训练：

- 低速移动中的外部触发；
- 原地或近似原地的小跳；
- 双轮离地；
- 软着陆；
- 着陆后恢复平衡。

第一版不直接训练障碍识别和自主决定起跳时机。`jump_trigger` 由命令生成器提供，后续可由键盘或高层规划器替换。

主要代码：

```text
source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/
└── wheel_legged_robot/
    ├── wheel_legged_jump_env_cfg.py
    ├── wheel_legged_flat_env_cfg.py
    ├── mdp/jump.py
    ├── agents/rsl_rl_ppo_cfg.py
    └── __init__.py
```

## 2. 为什么使用独立任务和统一策略

平地任务继续作为稳定回归基线，不在原任务中直接叠加跳跃奖励。跳跃任务从平地 checkpoint 初始化，然后训练一个同时处理移动和跳跃的统一低层策略。

统一策略的好处是起跳和落地之间没有策略硬切换。高层状态机决定当前阶段，并把阶段作为观测交给策略。当前 Stage B1 对左右腿长使用“阶段参考 + 策略残差”：

```text
最终腿长动作 = 阶段参考动作 + 0.15 × 策略腿长动作
```

轮速和腿角仍完全由策略控制。这样先把已由开环实验验证的下蹲—蹬伸—收腿物理动作交给参考轨迹，PPO 学习左右修正、姿态、轮速、落地和恢复，而不是从随机噪声中搜索一个只有约 0.2 s 的蹬伸序列。训练稳定后可逐步把残差比例提高，最后关闭参考辅助。

动作含义保持不变：

```text
[theta_L, length_L, wheel_vel_L, theta_R, length_R, wheel_vel_R]
```

状态机不会直接写电机力矩，也不会用固定轨迹替代策略。它只提供命令、阶段、奖励掩码和成功判定。

## 3. 跳跃命令

独立命令 `jump_command` 的形状为：

```text
[trigger, target_height, target_distance]
```

当前范围：

| 参数 | 数值 |
|---|---:|
| 命令周期 | 3–4 s |
| 有跳跃的周期比例 | 90% |
| 周期内触发延迟 | 0.6–1.0 s |
| trigger 脉冲宽度 | 0.10 s |
| 目标上升高度 | 0.04–0.06 m |
| 目标前向距离 | 0 m |

保留无跳跃周期的原因是防止策略忘记普通平衡和移动。速度命令当前限制在 `vx ±0.05 m/s`、`wz ±0.10 rad/s`，先降低起跳搜索难度。

## 4. 六阶段状态机

状态机每个控制步在物理更新后、奖励和终止计算前更新，因此奖励看到的是最新接触事件和最新阶段。

```text
IDLE
  │ trigger + 双轮接触 + 姿态/速度允许
  ▼
CROUCH ──实际平均腿长≤0.255 m──▶ THRUST
  ▲                                  │
  │                                  ├─蹬伸≥0.20 s、L0≥0.31 m、vz≥0.20──▶ FLIGHT
  │                                  │
  │                                  └─进入 FLIGHT 后 0.15 s 未离地─▶ RECOVERY（失败）
  │
  └────────────── RECOVERY ◀── LANDING ◀── FLIGHT
                    │             ▲            │
                    │             └─接地确认 2 步
                    │
                    ├─稳定 0.50 s 且高度达标─▶ IDLE（成功）
                    ├─稳定但高度未达标────────▶ IDLE（失败）
                    └─1.50 s 未恢复───────────▶ IDLE（失败）
```

各阶段目标：

| 阶段 | 主要职责 | 初始引导 |
|---|---|---:|
| IDLE | 平衡、低速跟踪、等待触发 | 普通高度命令 |
| CROUCH | 储存腿部弹性/做功行程 | `L0=0.22 m` |
| THRUST | 快速伸腿并形成起跳速度 | `L0=0.30 m` |
| FLIGHT | 保持姿态、准备接地 | `L0≈0.22 m` |
| LANDING | 吸收冲击 | `L0≈0.22 m` |
| RECOVERY | 恢复双轮接触和稳定平衡 | 普通平衡 |

关键阈值：

- 接触力阈值：`2 N`；
- 离地和着地均连续确认 `2` 个控制步；
- CROUCH 最长 `0.30 s`，没有真实蹲到 `0.255 m` 就判失败；
- THRUST 至少保持 `0.20 s`，避免轮端尚未卸载便提前收腿；
- 必须在 FLIGHT 阶段检测到双轮离地且 `vz>0.10 m/s`，才记录真实起跳；
- 开始跳跃要求倾角 `<0.30 rad`、`|vx|<0.40 m/s`；
- 恢复要求倾角 `<0.20 rad`、`|vz|<0.30 m/s`；
- 成功高度要求：实际 apex rise 至少达到目标高度的 `80%`。

最后一项很重要：只要短暂离地就给成功奖励会产生“极小弹跳”奖励漏洞。

## 5. 观测设计

平地 Actor/Critic 观测分别是 `36/153` 维。跳跃任务在末尾追加相同的 12 维：

| 新观测 | 维数 | 含义 |
|---|---:|---|
| `jump_phase` | 6 | 六阶段 one-hot |
| `wheel_contacts` | 2 | 左右轮二值接触 |
| `jump_command` | 3 | trigger、目标高度、目标距离 |
| `jump_phase_time` | 1 | 当前阶段时间，截断至 1 s |
| 合计 | 12 |  |

最终：

```text
Actor:  36  + 12 = 48
Critic: 153 + 12 = 165
```

采用 one-hot 而不是单个阶段编号，是为了避免网络误以为相邻数字代表相近物理状态。左右接触分开提供，后续可增加左右轮着地时间差奖励。

## 6. 分阶段奖励

RewardManager 会将配置权重乘以 `step_dt=0.01 s`。因此事件奖励的配置权重看起来较大，但单次实际贡献仍是有限的。

### 6.1 从平地任务继承并修改的奖励

| 奖励 | 权重 | 跳跃任务中的处理 |
|---|---:|---|
| `lin_vel_z` | -0.5 | 只在 IDLE/LANDING/RECOVERY 生效 |
| `base_height` | +2.5 | 只在 IDLE/RECOVERY 生效 |
| `orientation` | -4.0 | 全程保持，防止跳跃变成倾倒 |
| `dof_acc` | -5e-9 | 比平地更弱，允许快速推蹬 |
| `torques` | -5e-5 | 比平地更弱，但仍抑制无效大力矩 |
| `action_rate` | -0.001 | 放松快速动作约束 |
| `action_smooth` | -0.001 | 只保留轻微抑制 |
| `track_lin_vel` | +3.0 | 保留移动能力 |
| `track_ang_vel` | +1.0 | 保留转向能力 |
| `termination` | -100 | 终止事件约贡献 -1 |

如果不对 `lin_vel_z` 和 `base_height` 做阶段掩码，策略会同时收到“向上加速”和“不要产生竖直速度/不要离开高度”的冲突目标。

### 6.2 跳跃专用奖励

| 奖励 | 权重 | 生效阶段/事件 | 目的 |
|---|---:|---|---|
| `jump_crouch` | +8 | CROUCH | 跟踪 `0.22 m` 下蹲腿长 |
| `jump_phase_action` | +6 | CROUCH/THRUST/FLIGHT/LANDING | 让策略腿长残差配合阶段参考 |
| `jump_thrust_pose` | +5 | THRUST | 引导快速伸向 `0.30 m` |
| `jump_thrust_speed` | +5 | THRUST | 奖励实际腿长快速增长 |
| `jump_takeoff` | +8 | THRUST | 稠密奖励正向 `vz` 与弹道目标 |
| `jump_takeoff_event` | +100 | 首次真实离地 | 只奖励带正 `vz` 的确认离地 |
| `jump_height` | +6 | FLIGHT | 奖励高度进展并跟踪目标 apex rise |
| `jump_airborne` | +2 | FLIGHT | 确认真离地并提供稠密梯度 |
| `jump_symmetry` | +0.5 | CROUCH–LANDING | 左右腿长度和角度协调 |
| `jump_landing_pose` | +1 | FLIGHT/LANDING | 准备缓冲腿长 |
| `jump_landing_soft` | +40 | 首次着地事件 | 奖励小落地速度和小倾角 |
| `jump_landing_impact` | -15 | 首次着地事件 | 惩罚落地 `vz²` |
| `jump_recovery` | +1.5 | 真离地后的 RECOVERY | 双轮着地、竖直速度小、姿态直立 |
| `jump_success` | +200 | 成功事件 | 实际单次约 +2 |
| `jump_failure` | -100 | 失败事件 | 实际单次约 -1 |

腿长姿态奖励不是最终目标，而是训练早期的搜索引导。真正的性能目标仍然是起跳速度、目标高度、软着陆和恢复。策略稳定进入 FLIGHT 后，可以逐步把 `jump_crouch`、`jump_thrust_pose`、`jump_landing_pose` 降低到当前值的 30%–50%，增加动作自由度。

## 7. 第一轮训练停滞的根因与修复

第一轮 `2026-07-27_13-26-33_stage_b_small_jump` 并非单纯“奖励不够大”，而是存在四个相互强化的问题：

1. **假离地**：原状态机在 CROUCH 就累计无接触步数，下蹲卸载时可能直接被当成起跳；日志中的起跳 `vz<0` 是直接证据。
2. **奖励漏洞**：未真正离地的 RECOVERY 仍能持续获得恢复和对称奖励，策略学会“小蹬一下然后稳定”，而不是完成跳跃。
3. **探索噪声发散**：两个腿长动作的标准差增长到约 `16–17`，动作长期被裁剪到极限，PPO 实际无法精细控制腿长。
4. **搜索问题**：绝对动作策略需要偶然发现“蹲到位—约 0.20 s 快速蹬伸—离地后收腿”的短时序。即使修复状态机和噪声后，独立趋势测试中实际腿长仍停在约 `0.27 m`，说明仅继续堆训练步数不会解决问题。

对应修复：

- 每次进入新阶段清零该阶段的接触计数；
- 只有 FLIGHT 中连续两步双轮无接触且 `vz>0.10 m/s` 才确认起跳；
- 未真实离地时关闭 recovery 奖励，成功还要求 apex 达到目标的 80%；
- 训练期间把动作标准差约束在 `[0.05, 0.50]`；
- Stage B1 改为阶段腿长参考加 15% 策略残差；
- 延迟到完整蹬伸后再切换飞行收腿，避免在轮端卸载前抵消向上冲量。

修复后的验证结果：

| 验证 | 结果 |
|---|---:|
| 开环 + 平衡策略 apex rise | `0.0809 m` |
| 最大上升速度 | `0.666 m/s` |
| 连续双轮离地时间 | `0.080 s` |
| 短程 PPO（8 iter）J高度 | 约 `0.045 m` |
| 短程 PPO 动作 std | `0.26`，未发散 |

终端中的 `J起速/J落速/J高度` 是跨所有环境、所有采样步的平均统计，不是仅对起跳事件的条件均值。因此在触发占比较低时，`J起速=0.02` 不代表单次起跳只有 `0.02 m/s`。判断物理能力应同时查看 `J空`、`J落`、`J高度`、成功率和开环事件数据。

**不要从第一轮 `stage_b_small_jump` 的 checkpoint 续训。** 其中 Actor 和超大的探索标准差已经适应旧奖励漏洞，应重新从转换后的平地 checkpoint 启动。

## 8. 从平地 checkpoint 初始化

旧 checkpoint 的输入维度不同，不能直接加载。转换工具：

```text
scripts/expand_rsl_checkpoint_for_jump.py
```

转换时：

- 确定性策略的 Actor/Critic 权重保持不变；
- 首层末尾添加 12 个全零权重列；
- 新观测归一化均值初始化为 0；
- 新观测归一化方差和标准差初始化为 1；
- 默认将继承的高斯探索标准差上限设为 `0.30`；
- 原 checkpoint 不会被覆盖。

当前平地 `model_1999.pt` 的两个腿长动作标准差约为 `19.69/20.26`。这不会影响确定性 `play`，但 PPO 采样时会使腿长动作几乎总被裁剪到极限，是迁移跳跃训练必须处理的问题。转换后六维标准差为：

```text
[0.1819, 0.3000, 0.2985, 0.1647, 0.3000, 0.3000]
```

这里修改的是训练探索噪声，不是策略输出均值。若未来 checkpoint 的标准差已经合理，可显式传入 `--max-action-std 0` 禁用限制。

示例：

```bash
python scripts/expand_rsl_checkpoint_for_jump.py \
  logs/rsl_rl/wheel_legged_flat/<RUN>/model_1999.pt \
  --output logs/rsl_rl/wheel_legged_jump_flat/pretrained_flat/model_flat_jump_init.pt
```

开始训练：

```bash
python scripts/rsl_rl/train.py \
  --task Wheel-Legged-Jump-Flat-v0 \
  --headless \
  --num_envs 4096 \
  --resume \
  --load_run pretrained_flat \
  --checkpoint model_flat_jump_init.pt \
  --load_weights_only \
  --run_name stage_b_residual_v2
```

`--load_weights_only` 会载入 Actor/Critic，但不载入旧 Adam 状态，也不延续旧迭代编号。这是从平地任务迁移到新任务时应使用的模式。

## 9. 推荐训练阶段

### Stage B1：先学会执行阶段（约 300–800 iterations）

保持当前配置：

```text
vx:             ±0.05 m/s
wz:             ±0.10 rad/s
target_height:  0.04–0.06 m
jump_probability: 0.90
leg reference residual scale: 0.15
push:           disabled
```

通过标准：

- `jump_phase_flight` 不再长期为 0；
- 条件化的 `jump_takeoff_vz` 达到约 `0.5–0.8 m/s`；
- `jump_apex_rise` 随目标高度变化；
- `jump_success_rate > 0.50`；
- `joint_margin_min > 0`；
- `torque_saturation` 不长期上升。

不要仅根据总 reward 判断是否学会跳跃。

### Stage B2：重点训练落地（约 500–1000 iterations）

Stage B1 在 2000 iterations 后的完整趋势为：

```text
success_rate: 约 0.40（500 iter 后基本平台）
旧 J高度:     约 0.056 m
旧 J起速:     约 0.078（被零值稀释）
```

旧 `J高度` 包含轮子仍接地时伸腿造成的机身升高，且成功判定没有最低
滞空时间，因此 0.40 不是继续训练可以突破的优化平台，而是浅蹲参考和
旧统计口径共同形成的局部最优。

当前 Stage B2 使用：

```text
crouch reference:       0.19 m
crouch ready threshold: 0.22 m
minimum crouch time:    0.22 s
flight/landing length:  0.19 m
leg residual scale:     0.05
target base rise:       0.05–0.07 m
minimum real air time:  0.12 s
```

成功必须同时满足：确认双轮离地、滞空至少 `0.12 s`、机身上升达到目标
的 80%、落地并恢复稳定。`J轮高` 作为诊断保留，但在加入地形/障碍物
高度采样前不作为成功门槛。

从 Stage B1 的 `model_2000.pt` 只加载权重开始 B2：

```bash
python scripts/rsl_rl/train.py \
  --task Wheel-Legged-Jump-Flat-v0 \
  --headless \
  --num_envs 4096 \
  --resume \
  --load_run 2026-07-27_15-44-15_stage_b_residual_v2 \
  --checkpoint model_2000.pt \
  --load_weights_only \
  --run_name stage_b2_high_jump
```

开环验证达到约 `0.10 m` 机身上升、`1.26 m/s` 最大竖直速度和
`0.20 s` 双轮离地。512 环境短程迁移验证中，条件化起跳速度约
`0.53 m/s`、滞空约 `0.15 s`，成功率曾达到 `0.75–0.79`。初始落地
速度约 `-1.03 m/s`，因此 B2 的主要学习目标是降低冲击并稳定恢复。

当成功离地但落地不稳时：

- 暂不增加目标高度；
- 保持或适当提高 `jump_landing_impact`；
- 提高恢复样本比例；
- 检查 `jump_landing_vz`、倾角和着陆后存活时间；
- 若策略已可靠按阶段改变腿长，将三个姿态引导奖励逐步减半。

通过标准：

```text
success_rate > 0.75
|landing_vz| 的中位数 < 0.8 m/s
着陆后稳定存活 > 0.5 s
```

### Stage C：移动中小跳

逐步把速度范围扩大为：

```text
±0.20 → ±0.40 → ±0.70 → ±1.00 m/s
```

每次只扩一个等级，并同时保留无跳跃周期。目标不是一开始就高速越障，而是保证：

- 起跳前速度跟踪；
- 飞行中俯仰可控；
- 落地后恢复原速度命令；
- 无 trigger 时性能没有明显退化。

### Stage D：目标距离和障碍

当前 `target_distance=0`。开始前向跳跃后再加入：

- 起跳点到落地点的水平位移跟踪；
- 障碍物碰撞惩罚；
- 越过障碍事件；
- 落地后恢复事件；
- 高层提供的 trigger/目标高度/目标距离。

不要仅奖励 base 高度，否则策略可能原地竖跳而不越障，或通过倾倒抬高 base。

## 10. 训练监控

终端调试行已经加入：

```text
J蹲 J蹬 J空 J落 J稳 J目标 J起速 J落速 J高度 J成功
```

分别表示各阶段环境比例、平均目标高度、记录的起跳/落地速度、apex rise 和累计成功率。

典型判断：

| 表现 | 可能原因 | 下一步 |
|---|---|---|
| 有 `J蹲/J蹬`，无 `J空` | 推蹬没有形成离地 | 先看腿长和起跳 `vz`，不要先提高成功奖励 |
| `J空` 增长但 `J成功=0` | 高度不足或恢复失败 | 分别看 `J高度`、`J落速` |
| 高度达标，落地经常终止 | 缓冲能力不足 | 保持高度范围，强化 landing/recovery |
| 普通移动明显退化 | 跳跃样本占比过高 | 降低 jump probability 或增加无跳跃周期 |
| 力矩长期饱和 | 轨迹/增益超出执行能力 | 降低目标高度或检查 VMC，不要靠 PPO 硬顶 |

## 11. 已完成验证

实现后已完成：

1. Python 编译检查；
2. 新任务 Gym 注册与配置解析；
3. 128 环境、8 iterations 的新策略冒烟训练；
4. checkpoint 36/153 → 48/165 转换；
5. 从转换后的平地模型只加载权重并完成 PPO 更新；
6. 用已知可离地轨迹验证真实接触与完整六阶段状态机；
7. 验证阶段参考 + 策略残差能够在短程 PPO 中覆盖 CROUCH、THRUST、FLIGHT、LANDING 和 RECOVERY。

完整状态机验证结果保存在：

```text
jump_open_loop_results/jump_task_residual_validation_v5/
```

使用 `0.22 m / 0.15 s → 0.30 m` 轨迹时：

```text
apex rise:       0.0809 m
max vz:          0.666 m/s
air time:        0.08 s
landing vz:     -0.644 m/s
max tilt:        0.196 rad
torque clipping: 0
joint margin:    > 0
```

`summary.csv` 确认 IDLE、CROUCH、THRUST、FLIGHT、LANDING、RECOVERY 六个阶段均实际到达。由此同时验证了 trigger 时序、双轮接触判定、离地/着地确认以及阶段转换链路。

## 12. Stage C1：0.09--0.12 m 高跳与落地强化

新任务 ID：

```text
Wheel-Legged-Jump-High-Landing-Flat-v0
```

该任务保留 Stage B2 的观测维度、六维动作和六阶段编号，因此可以直接加载
Stage B2 checkpoint。它不会修改原来的 `Wheel-Legged-Jump-Flat-v0`。

主要变化：

- 跳跃目标提高到 `0.09--0.12 m`；
- 目标越高，下蹲参考由 `0.19 m` 连续加深到 `0.18 m`；
- 上升阶段保持 `0.19 m` 收腿；
- apex 前随垂直速度连续预伸到 `0.30 m`；
- 触地后在 `0.20 s` 内连续压缩到 `0.21 m`，同时提高虚拟腿阻尼；
- 高跳阶段提高蹬伸刚度和前馈，但不改变普通移动阶段增益；
- 成功要求落地速度不超过 `1.05 m/s`，同时单独记录严格
  `|J落速| <= 0.8 m/s` 的 `J柔落` 比率。

推荐从 Stage B2 的 `model_700.pt` 只迁移网络权重：

```bash
cd Wheel-Legged-Lab

python scripts/rsl_rl/train.py \
  --task Wheel-Legged-Jump-High-Landing-Flat-v0 \
  --headless \
  --load_checkpoint_path \
  logs/rsl_rl/wheel_legged_jump_flat/2026-07-27_23-20-42_stage_b2_high_jump/model_700.pt \
  --load_weights_only \
  --run_name stage_c1_high_landing
```

训练中建议每 `100` 轮保存并在 `model_300/500/700/...` 上做确定性
Play，不要只保留最后一个模型：

```bash
python scripts/rsl_rl/play.py \
  --task Wheel-Legged-Jump-High-Landing-Flat-v0 \
  --checkpoint \
  logs/rsl_rl/wheel_legged_jump_high_landing_flat/<run>/model_500.pt \
  --num_envs 50 \
  --command_range 0.05 \
  --yaw_command_range 0.10
```

阶段验收建议：

```text
J机身升 >= 0.09 m
J空时   >= 0.18 s
J成功   >= 0.75（随后目标 0.85）
J柔落   >= 0.60（随后目标 0.80）
|J落速| 的条件平均值 <= 0.9 m/s，最终 <= 0.8 m/s
F恢复   <= 0.03
tau_clip 接近 0
```

不要在这一阶段扩大 `vx`。达到上述条件后，以本阶段最佳 checkpoint
建立 Stage C2 移动跳跃，再逐级扩大速度和目标距离。

## 13. Stage C2：轮端 0.08--0.12 m 净空强化

新任务 ID：

```text
Wheel-Legged-Jump-Clearance-Flat-v0
```

Stage C1 的 `J机身升` 约为 0.11 m，但 `J轮高` 只有约 0.05 m。根因并
不是起跳完全不足，而是原来的下降准备从 `vz=+0.30 m/s` 就开始，且整个
FLIGHT 阶段仍保留 55% 支撑前馈；轮腿在到达 apex 前已经重新伸长。因此，
机身高度不能代表能越过多高的障碍。

本阶段不修改 Stage C1，而是增加独立任务，并做以下调整：

- 奖励直接测量两只轮子中较低者相对起跳位置的真实净空；
- 成功同时要求机身上升、连续腾空、轮端净空和安全落地；
- 上升与 apex 附近保持 `0.18 m` 收腿参考，并关闭支撑前馈；
- 收腿阶段单独提高刚度、降低阻尼，使短暂飞行窗口内能完成收腿；
- 到 `vz < -0.08 m/s` 后才开始预伸腿，随后在下降阶段伸到 `0.30 m`；
- 触地后继续使用 Stage C1 的高阻尼吸能和恢复控制；
- 训练目标均匀覆盖 `0.08--0.12 m`，而不是只记忆一个固定高度。

从目前 Stage C1 最佳模型迁移：

```bash
cd Wheel-Legged-Lab

python scripts/rsl_rl/train.py \
  --task Wheel-Legged-Jump-Clearance-Flat-v0 \
  --headless \
  --load_checkpoint_path \
  logs/rsl_rl/wheel_legged_jump_high_landing_flat/2026-07-28_09-46-30_stage_c1_high_landing/model_1000.pt \
  --load_weights_only \
  --run_name stage_c2_wheel_clearance
```

固定用 `0.12 m` 命令做 Play 验收：

```bash
python scripts/rsl_rl/play.py \
  --task Wheel-Legged-Jump-Clearance-Flat-v0 \
  --checkpoint \
  logs/rsl_rl/wheel_legged_jump_clearance_flat/<run>/<model>.pt \
  --num_envs 50 \
  --command_range 0.05 \
  --yaw_command_range 0.10 \
  --jump_height 0.12
```

初始迁移短测（64 环境、5 iterations）已经把平均 `J轮高` 从 Stage C1
的约 `0.05 m` 提高到约 `0.10 m`，同时得到：

```text
J机身升 = 0.105 m
J轮高   = 0.101 m
J空时   = 0.156 s
J落速   = -0.729 m/s
J柔落   = 0.809
tau_clip = 0
```

这只是旧策略直接迁移后的起点，并非训练完成结果。正式验收应固定
`--jump_height 0.12`，并要求：

```text
J轮高 >= 0.115 m（最终目标 >= 0.120 m）
J成功 >= 0.75（最终目标 >= 0.85）
J柔落 >= 0.70
|J落速| <= 0.80 m/s
F卸 <= 0.05
F恢复 <= 0.03
tau_clip 接近 0
```

建议比较 `model_300.pt`、`model_500.pt`、`model_700.pt`，以固定
`0.12 m` Play 的综合指标选择最佳模型，而不是只按训练总奖励或最后一个
checkpoint 选择。达到本阶段验收后再建立移动跳跃任务；不要在净空尚未
稳定时同时扩大 `vx`。

## 14. Stage C3-L1：±0.20 m/s 移动跳跃

新任务 ID：

```text
Wheel-Legged-Jump-Moving-Flat-v0
```

进入依据来自 Stage C2 最新 Play：

```text
J轮高       约 0.107--0.110 m
J成功       稳态约 0.83--0.94
J柔落       稳态约 0.93--1.00
J落速       约 -0.69 m/s
F卸/F恢复   0
tau_clip    0
```

这已经满足移动跳跃第一档的进入条件。C3-L1 只增加一个难度维度：

```text
vx = [-0.20, 0.20] m/s
wz = [-0.20, 0.20] rad/s
```

轮端高度仍使用 Stage C2 的 `0.08--0.12 m`，不改变收腿、预伸和落地
吸能轨迹，也暂不训练目标距离。

实现要点：

- 跳跃触发时锁存 `vx`，整个跳跃都以该值作为速度目标；
- 状态机非 IDLE 时禁止速度命令重新采样，避免飞行中目标突变；
- 分别奖励起跳前速度跟踪、空中水平速度保持和首次触地速度；
- 成功恢复时要求 `|vx - target_vx| <= 0.15 m/s`；
- 新增 `J目标vx`、`J起vx误差`、`J落vx误差` 调试指标；
- jump probability 从 `0.90` 降至 `0.80`，保留足够无跳跃移动样本。

从 Stage C2 最新 checkpoint 迁移：

```bash
cd Wheel-Legged-Lab

python scripts/rsl_rl/train.py \
  --task Wheel-Legged-Jump-Moving-Flat-v0 \
  --headless \
  --load_checkpoint_path \
  logs/rsl_rl/wheel_legged_jump_clearance_flat/2026-07-28_12-26-32_stage_c2_wheel_clearance/model_200.pt \
  --load_weights_only \
  --run_name stage_c3_l1_moving_020
```

Play：

```bash
python scripts/rsl_rl/play.py \
  --task Wheel-Legged-Jump-Moving-Flat-v0 \
  --checkpoint \
  logs/rsl_rl/wheel_legged_jump_moving_flat/<run>/<model>.pt \
  --num_envs 50 \
  --command_range 0.20 \
  --yaw_command_range 0.20 \
  --jump_height 0.10
```

64 环境、5 iterations 迁移冒烟测试结果：

```text
J轮高       0.1097 m
J成功       0.8609
J柔落       0.9671
J落速      -0.6946 m/s
J起vx误差   0.1428 m/s
J落vx误差   0.0898 m/s
vx sign     0.8549
F卸/F恢复   0
tau_clip    0
```

垂直跳跃能力没有因速度范围扩大而立即退化。正式训练的主要任务是把
起跳速度误差降下来。建议验收：

```text
J轮高             >= 0.10 m
J成功             >= 0.80
J柔落             >= 0.80
J起vx误差          <= 0.08 m/s
J落vx误差          <= 0.08 m/s
vx_tracking_gain   0.85--1.15
vx_sign_match      >= 0.90
F卸/F恢复          <= 0.03
tau_clip           接近 0
```

达到以上指标后，再建立 C3-L2，将速度扩大到 `±0.40 m/s`。不要直接在
C3-L1 内把范围改到 `±1.0 m/s`，也不要同时引入目标距离和实体障碍物。

## 15. Stage C3-Auto：航向对齐与 0.2→1.0 m/s 自动课程

新任务 ID：

```text
Wheel-Legged-Jump-Moving-Curriculum-Flat-v0
```

### 航向不对齐的根因

C3-L1 同时存在两个冲突目标：

```text
heading_command = False
wz 随机范围     = [-0.20, 0.20]
heading 目标     = 0
```

策略被角速度命令要求持续转向，但 `track_heading` 又要求回到世界 yaw=0。
训练日志中的 `|wz|≈0.10`、`heading error≈0.65 rad` 和
`gain_wz≈0.5` 正是这一冲突的表现，不是单纯增加航向奖励就能解决。

新任务改为：

```text
heading_command = True
heading target  = 0 rad
heading stiffness = 0.8
wz clamp = [-1.2, 1.2] rad/s
```

此时 `wz` 不再随机采样，而是由航向误差自动生成纠偏命令。航向位置奖励
和角速度跟踪奖励因此指向同一目标。短程迁移测试中，旧模型未重新训练时
`heading error` 已从约 `0.65 rad` 连续降至约 `0.12 rad`，同时保持：

```text
J轮高   约 0.109 m
J成功   约 0.86--0.91
J柔落   约 0.95--1.00
F卸/F恢复 = 0
```

### 自动速度课程

同一次训练依次使用：

```text
±0.20 → ±0.40 → ±0.60 → ±0.80 → ±1.00 m/s
```

每个统计窗口至少包含 512 个完整 episode 和 1024 次跳跃尝试。以下四项
必须同时达标，并连续通过两个独立窗口，才会升级：

```text
移动跟踪课程得分 >= 0.78
跳跃成功率       >= 0.75
柔落率           >= 0.75
航向课程得分     >= 0.80
```

因此不会出现“只学会高速移动、跳跃已经退化却仍升级”的情况。速度升级后
继续保留完整 `[-limit,+limit]` 采样和无跳跃周期，所以基础能力相同，但
高速起跳、空中姿态和落地水平动量仍需要逐档适应，不能把五档视为完全相同
的动力学问题。

新增终端指标：

```text
MJ档    0/1/2/3/4，对应 0.2/0.4/0.6/0.8/1.0 m/s
cur     移动跟踪窗口得分
MJ成功  课程窗口跳跃成功率
MJ柔落  课程窗口柔落率
MJ航向  课程窗口航向得分
MJ连过  当前连续通过窗口数
```

从 C3-L1 `model_700.pt` 迁移：

```bash
cd Wheel-Legged-Lab

python scripts/rsl_rl/train.py \
  --task Wheel-Legged-Jump-Moving-Curriculum-Flat-v0 \
  --headless \
  --load_checkpoint_path \
  logs/rsl_rl/wheel_legged_jump_moving_flat/2026-07-28_13-41-28_stage_c3_l1_moving_020/model_700.pt \
  --load_weights_only \
  --run_name stage_c3_auto_speed_heading
```

训练时观察 `vx_lim` 和 `MJ档` 即可确认实际升级。课程到达第 4 档后仍会
保持 `±1.0 m/s` 继续优化，不会越过上限。

最终随机速度 Play：

```bash
python scripts/rsl_rl/play.py \
  --task Wheel-Legged-Jump-Moving-Curriculum-Flat-v0 \
  --checkpoint \
  logs/rsl_rl/wheel_legged_jump_moving_curriculum_flat/<run>/<model>.pt \
  --num_envs 50 \
  --command_range 1.0 \
  --yaw_command_range 1.2 \
  --jump_height 0.10
```

Play 会禁用速度课程并服从 `--command_range`，不会重新从 0.2 m/s 开始。

### 从指定课程档位恢复

课程状态不包含在 RSL-RL 网络 checkpoint 中。若训练中断后直接
`--load_weights_only`，环境默认会从第 0 档重新开始。训练脚本因此提供：

```text
--moving_jump_initial_level {0,1,2,3,4}
```

例如第 3 档已经稳定、但旧的 `0.80` 跟踪阈值使课程停滞时，可保留当前
模型权重并从 `±0.80 m/s` 继续：

```bash
python scripts/rsl_rl/train.py \
  --task Wheel-Legged-Jump-Moving-Curriculum-Flat-v0 \
  --headless \
  --load_checkpoint_path \
  logs/rsl_rl/wheel_legged_jump_moving_curriculum_flat/2026-07-28_14-35-18_stage_c3_auto_speed_heading/model_900.pt \
  --load_weights_only \
  --moving_jump_initial_level 3 \
  --run_name stage_c3_resume_level3_to_1ms
```

统一课程跟踪阈值现为 `0.78`；跳跃成功、柔落、航向阈值以及连续通过两个
窗口的规则保持不变。

### 键盘自由移动与跳跃

RSL-RL Play 的 `--keyboard` 模式支持：

```text
↑ / ↓       前进 / 后退
Z / X       左转 / 右转
R / F       升高 / 降低机身
L           清零移动命令并恢复默认高度
J           触发一次跳跃
```

键盘模式会关闭随机跳跃。每次按下 J 会输出与训练一致时长的 trigger
脉冲；当前跳跃尚未恢复时再次按下会被忽略，不会在落地后意外补跳。

```bash
python scripts/rsl_rl/play.py \
  --task Wheel-Legged-Jump-Moving-Curriculum-Flat-v0 \
  --checkpoint \
  logs/rsl_rl/wheel_legged_jump_moving_curriculum_flat/2026-07-28_15-31-13_stage_c3_resume_level3_to_1ms/model_200.pt \
  --keyboard \
  --real-time \
  --command_range 1.0 \
  --yaw_command_range 1.2 \
  --jump_height 0.10
```

不要添加 `--headless`；启动后先点击一次仿真 Viewport，使键盘焦点进入窗口。

### 键盘复杂地形测试

键盘 Play 现在支持单机器人测试地形，不会改动训练配置。使用：

```text
--keyboard_terrain {flat,slope,stairs,mixed}
--keyboard_terrain_difficulty 0.0~1.0
```

- `flat`：原有平地，默认值；
- `slope`：中心平台和四周斜坡；
- `stairs`：中心平台和阶梯；
- `mixed`：斜坡、阶梯和粗糙地面随机选一种；
- 难度默认 `0.5`，控制斜坡角度和阶梯高度。

复杂地形参数必须与 `--keyboard` 一起使用，避免在批量 Play 中误生成只有一个
地形岛的场景。例如：

```bash
python scripts/rsl_rl/play.py \
  --task Wheel-Legged-Jump-Moving-Curriculum-Flat-v0 \
  --checkpoint /absolute/path/to/model.pt \
  --keyboard \
  --keyboard_terrain stairs \
  --keyboard_terrain_difficulty 0.5 \
  --command_range 1.0 \
  --yaw_command_range 1.2 \
  --real-time
```

场景大小约为 `12 m × 12 m`，中央为安全平整出生平台。建议从难度 `0.3` 开始，
确认机器人能够保持平衡并上下坡/阶梯后，再测试 `0.7~1.0`。这类测试主要用于
观察平衡、速度保持、机身高度调节和跳跃恢复，不代表模型已经针对复杂地形训练
完成；如果需要在复杂地形上稳定运行，应另建对应的训练环境和地形课程。

## 16. Stage D2：Oracle 实体障碍物细粒度课程与固定尺寸评估

> 更新日期：2026-07-31

任务 ID：

```text
Wheel-Legged-Jump-Obstacle-Oracle-Flat-v0
```

### Play 为什么默认显示第 0 档

RSL-RL checkpoint 保存 actor、critic、优化器等训练状态，但不会保存环境内部的
障碍物课程档位。每次新建 Play 环境时，几何课程因此从配置中的
`initial_level=0` 初始化。未指定固定尺寸时，日志显示：

```text
O高=0.0200 O宽=0.0350 O档=0
```

这是新环境的正常初始化结果，不表示加载的策略权重只能通过第 0 档。评估不同
尺寸时不应依赖刚创建的课程自行晋级，而应显式指定障碍物高度和宽度。

Play 新增两个参数：

```text
--obstacle_height HEIGHT   障碍物暴露高度，当前允许 0–0.08 m
--obstacle_width WIDTH     沿行驶方向的障碍物宽度，必须大于 0
```

例如精确测试最终档 8 cm 高、8 cm 宽障碍物：

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Wheel-Legged-Jump-Obstacle-Oracle-Flat-v0 \
  --checkpoint /absolute/path/to/obstacle_oracle/model_xxx.pt \
  --num_envs 10 \
  --command_range 0.75 \
  --yaw_command_range 0.4 \
  --obstacle_height 0.08 \
  --obstacle_width 0.08
```

固定几何模式下，环境不再对所给高度或宽度进行课程随机化。紧凑日志使用
`O档=-1` 标识手动固定尺寸，此时应以 `O高` 和 `O宽` 为准。只给出其中一个
参数时，另一个维度仍采用新环境第 0 档的配置；正式对比测试建议两个参数同时
给出。

### 七档障碍物几何课程

原来的 `2→4→6→8 cm` 四档课程在后两档同时提高高度和宽度，难度跃迁过大。
当前课程拆分为七档：

| `O档` | 随机高度上限 | 固定宽度 |
|---:|---:|---:|
| 0 | 0.02 m | 0.035 m |
| 1 | 0.04 m | 0.035 m |
| 2 | 0.05 m | 0.050 m |
| 3 | 0.06 m | 0.050 m |
| 4 | 0.07 m | 0.065 m |
| 5 | 0.08 m | 0.065 m |
| 6 | 0.08 m | 0.080 m |

训练随机高度仍从“当前档上限以下 1 cm”到“当前档上限”采样，最低不小于
2 cm。新表尽量避免一次晋级同时大幅增加高度和宽度：先形成轮端净空能力，
再增加障碍物上方需要保持净空的时间。

### 成功判定修正

障碍物成功仍必须满足真实腾空、轮端越障净空、干净跨越、落点、姿态和速度
恢复等条件。仅将机身上升比例门槛修改为：

```python
state.success_height_ratio = 0.40
```

该修改用于减少“轮腿已经通过收腿获得足够净空并跨过实体障碍物，但机身高度
只差几毫米而被记为失败”的假阴性。它不会绕过碰撞、净空或落地条件，因此
不能把真实碰撞误判为成功。

### 最高档前恢复探索

Stage D2 的 PPO 使用：

```text
entropy_coef = 1.5e-3
min_action_std = 0.10
max_action_std = 0.50
```

较高的熵奖励和动作标准差下限用于防止策略在前几档过早收敛，使其进入更高、
更宽的障碍物档位后仍能探索新的起跳、收腿和落地动作。标准差上下限由训练
脚本对 Oracle 障碍物任务默认安装，也可以用命令行显式覆盖。

课程状态不会随模型恢复，因此不建议从已经在旧最高档退化后的
`model_1799.pt` 继续。使用进入最高档前能力较稳定的 `model_1100.pt`，并从
新课程第 3 档（6 cm 高上限、5 cm 宽）重新适应：

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Wheel-Legged-Jump-Obstacle-Oracle-Flat-v0 \
  --headless \
  --num_envs 4096 \
  --load_checkpoint_path \
  logs/rsl_rl/wheel_legged_jump_obstacle_oracle_flat/2026-07-31_12-15-58_obstacle_curriculum_relaxed_gate/model_1100.pt \
  --load_weights_only \
  --obstacle_initial_level 3 \
  --min_action_std 0.10 \
  --max_action_std 0.50 \
  --run_name obstacle_curriculum_fine_7_levels
```

`--load_weights_only` 保留 actor/critic 能力，但重新创建优化器和 iteration
计数；`--obstacle_initial_level 3` 则显式恢复环境课程起点。训练时应同时检查
`O成功`、`O跨越`、`O碰撞` 和 `F性能`：成功定义修正后若 `F性能` 下降但碰撞
仍高，说明剩余问题是真实越障能力，而不是统计口径。

需要区分最终成功率和课程晋级门槛。最新第 5 档日志（约第 384--403 次
迭代）中，`O跨越` 约为 `0.78--0.79`，`O碰撞` 约为 `0.21--0.25`，已经满足
当前晋级的跨越和碰撞条件；但 `O成功` 约为 `0.34`，`O课成功` 约为
`0.36--0.43`，说明完整跳跃成功仍受落地/恢复和无碰撞条件限制。因此本轮只
将**课程晋级**的 `success_threshold` 从 `0.60` 放宽到 `0.45`，最终
`O成功` 的物理判定不改变，`clear_threshold=0.75`、`collision_threshold=0.25`
也不改变，并继续要求连续两个窗口通过。这样是放宽课程阻塞，而不是伪造
成功率；如果后续 `O成功` 仍长期低于 `0.40`，下一步应增加各个成功条件的
失败分解统计，而不是继续盲目降低门槛。

### 第 6 档约 0.5 成功率的针对性优化（2026-07-31）

最新第 6 档 iteration 377--402 的训练窗口并不是“所有指标都很好，只差统计
标准”：

```text
O课成功  平均约 0.483，最高约 0.511
O课跨越  平均约 0.791
O课碰撞  平均约 0.254
O成功    平均约 0.421
O碰撞    平均约 0.291，且几乎全部为轮端接触
J起vx误差 平均约 0.197 m/s
J落点成功 平均约 0.724
J柔落    平均约 0.607
```

因此不继续取消干净跨越或轮端净空条件。本轮采用以下最小修改：

1. 障碍物上方净空奖励 `18→28`，干净跨越奖励 `220→240`，首次实体碰撞
   惩罚 `-300→-360`，直接处理最终宽障碍物上的轮端接触；
2. 空中速度保持奖励 `6→10`、落地速度奖励 `80→100`，目标接近奖励
   `8→12`、落点事件奖励 `180→220`，改善约 `0.20 m/s` 的起跳水平速度
   误差和约 `0.72` 的落点成功率；
3. 障碍物远边后的落点余量 `0.04→0.06 m`，避免策略刚过远边就落轮；
4. Oracle 最终成功的恢复水平速度误差容差 `0.18→0.20 m/s`，落点容差
   `0.05→0.06 m`。这只减少边界落点/速度的假阴性，实体无碰撞、最小
   `1.5 cm` 轮端净空、真实腾空、落速和恢复仍为必需条件。

一键流水线的障碍物训练态验收按同一批日志设为：到达第 6 档、
`O课成功≥0.45`、`O课跨越≥0.75`、`O课碰撞≤0.28`，并同时要求累计
`O成功≥0.40`、`O跨越≥0.70`、`O碰撞≤0.30`。使用 20-iteration 均值连续
3 次通过，目标是允许训练噪声而不是放弃物理质量。
