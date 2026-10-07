# 轮腿机器人一键分阶段训练

## 1. 目标与完整阶段

流水线现在覆盖从基础平地到深度感知实体障碍物的全部十个阶段：

```text
flat
→ recovery
→ terrain_reactive
→ jump_flat
→ high_landing
→ clearance
→ moving_curriculum
→ target_landing
→ obstacle_oracle
→ obstacle_perceptive
```

对应任务：

```text
Wheel-Legged-Flat-v0
→ Wheel-Legged-Recovery-Flat-v0
→ Wheel-Legged-Terrain-Reactive-v0
→ Wheel-Legged-Jump-Flat-v0
→ Wheel-Legged-Jump-High-Landing-Flat-v0
→ Wheel-Legged-Jump-Clearance-Flat-v0
→ Wheel-Legged-Jump-Moving-Curriculum-Flat-v0
→ Wheel-Legged-Jump-Target-Landing-Flat-v0
→ Wheel-Legged-Jump-Obstacle-Oracle-Flat-v0
→ Wheel-Legged-Jump-Obstacle-Perceptive-Flat-v0
```

`recovery` 只处理可恢复倾角和推撞，不包含完全侧躺起立；
`terrain_reactive` 使用本体感觉适应坡面、低阶梯和粗糙地面。外层流水线负责
任务切换、checkpoint 迁移和自动验收；移动跳跃任务内部把
速度从 `0.2` 自动提高到 `1.0 m/s`，障碍物任务内部把几何从第 0 档提高到
第 6 档。

## 2. 从头训练或只训练到指定阶段

完整训练：

```bash
./scripts/rsl_rl/train_staged.sh \
  --num-envs 4096 \
  --device cuda:0 \
  --seed 42
```

新增的 `--end-stage` 表示“最后一个实际训练并验收的阶段”，包含该阶段。
例如只完成到轮端净空：

```bash
./scripts/rsl_rl/train_staged.sh --end-stage clearance
```

只检查命令和阶段范围，不启动训练：

```bash
./scripts/rsl_rl/train_staged.sh --dry-run --end-stage target_landing
```

未提供 `--end-stage` 时默认训练到配置中的最后阶段 `obstacle_perceptive`。

## 3. 从已有 checkpoint 开始

通用恢复入口由三个参数共同组成：

- `--start-checkpoint`：模型文件；
- `--start-stage`：该模型所属的阶段，不是准备进入的阶段；
- `--start-mode continue|next`：继续当前阶段，或确认当前阶段已合格并进入下一阶段。

例如从合格 Flat 模型进入 Recovery，并训练到 Clearance：

```bash
./scripts/rsl_rl/train_staged.sh \
  --start-checkpoint logs/rsl_rl/wheel_legged_flat/<run>/model_<iteration>.pt \
  --start-stage flat \
  --start-mode next \
  --end-stage clearance
```

流水线在 `terrain_reactive → jump_flat` 时自动调用
`expand_rsl_checkpoint_for_jump.py` 扩展 Actor、Critic 和观测归一化器。旧参数
`--flat-checkpoint PATH` 仍兼容，等价于 `flat + next`，即从 Recovery 开始。

继续训练尚未达标的第六档障碍物模型：

```bash
./scripts/rsl_rl/train_staged.sh \
  --start-checkpoint \
  logs/rsl_rl/wheel_legged_jump_obstacle_oracle_flat/<run>/model_400.pt \
  --start-stage obstacle_oracle \
  --start-mode continue \
  --end-stage obstacle_oracle
```

所有合法阶段名依次为：

```text
flat, recovery, terrain_reactive, jump_flat, high_landing,
clearance, moving_curriculum, target_landing, obstacle_oracle,
obstacle_perceptive
```

`continue` 恢复网络、优化器和 iteration，并继续执行当前阶段验收；`next`
只迁移 Actor/Critic，使用新优化器和新 iteration。`--end-stage` 不能早于
实际开始训练的阶段。例如 `--start-stage clearance --start-mode next`
实际从 `moving_curriculum` 开始，因此不能同时指定 `--end-stage clearance`。
最终阶段 `obstacle_perceptive` 后没有下一阶段，只能使用 `continue`。

## 4. 自动验收原则

所有阶段使用最近 `20` 个 iteration 的滑动平均，并要求连续 `3` 次通过。
相比原来的连续 `5` 次，这仍能过滤单次尖峰，但不容易因训练噪声长期卡住。
每个阶段还保留最低训练轮数和多项联合条件，不能只靠单个成功率晋级。

当前主要门槛如下；完整机器可读配置位于
`scripts/rsl_rl/staged_training_config.json`。

| 阶段 | 主要训练态验收条件 |
|---|---|
| Flat | 速度档 `≥0.99 m/s`、yaw 档 `≥1.99 rad/s`、线速度得分 `≥0.85`、角速度得分 `≥0.79` |
| Recovery | 扰动恢复成功率 `≥0.70`、失败率 `≤0.30`、速度跟踪增益 `≥0.55` |
| Terrain Reactive | 恢复成功率 `≥0.68`、命令方向行驶比 `≥0.65`、倾角 `≤0.35 rad` |
| Jump Flat | 起跳 `≥0.45 m/s`、机身上升 `≥0.055 m`、滞空 `≥0.125 s`、成功率 `≥0.62` |
| High Landing | 上升 `≥0.085 m`、成功率 `≥0.42`、柔落率 `≥0.70`、平均落速绝对值 `≤1.0 m/s` |
| Clearance | 轮端净空 `≥0.095 m`、成功率 `≥0.60`、柔落率 `≥0.75` |
| Moving Curriculum | 到第 4 档、移动得分 `≥0.75`、课程成功 `≥0.70`、柔落 `≥0.75`、航向 `≥0.76` |
| Target Landing | 成功率 `≥0.50`、落点成功 `≥0.68`、平均落点误差 `≤0.045 m`、柔落 `≥0.65` |
| Obstacle Oracle | 到第 6 档、课程成功 `≥0.45`、课程跨越 `≥0.75`、课程碰撞 `≤0.28`；累计成功 `≥0.40`、跨越 `≥0.70`、碰撞 `≤0.30` |
| Obstacle Perceptive | 到第 6 档、感知有效率 `≥0.70`、累计成功 `≥0.30`、跨越 `≥0.65`、碰撞 `≤0.40` |

各阶段仍检查 `torque_saturation ≤ 0.05`，跳跃阶段也保留恢复失败率限制。
这些是带探索噪声的训练态自动切换门槛，不等于最终确定性 Play 验收值。
障碍物门槛依据 2026-07-31 第 6 档日志校准：`O课成功≈0.48`、
`O课跨越≈0.79`、`O课碰撞≈0.25`、累计 `O成功≈0.42`。因此自动门槛允许
正常波动，但没有取消真实跨越、碰撞和安全条件。

## 5. Checkpoint 迁移

- Flat → Recovery → Terrain Reactive：Actor 维度一致，直接迁移网络权重。
- Terrain Reactive → Jump Flat：自动扩展跳跃观测输入，再只加载网络权重。
- 可选的 Terrain Perceptive Actor 多 15 维局部高度扫描，不属于默认流水线；
  从 Reactive 单独训练时由 `train.py` 自动扩展输入层。
- Jump Flat → Target Landing：观测和动作维度一致，直接迁移权重。
- Target Landing → Obstacle Oracle：障碍物观测多 10 维前向扫描；
  `train.py` 在 `--load_weights_only` 时自动扩展 checkpoint 输入层。
- 每次跨任务迁移使用新优化器，避免沿用上一奖励分布的 Adam 动量。
- `continue` 同阶段恢复不会执行权重-only迁移，而是恢复完整训练状态。

## 6. 日志、停止范围和失败保护

每次运行建立：

```text
logs/rsl_rl/staged_pipeline/<timestamp>/
```

其中包括：

- `state.json`：`first_training_stage`、`end_stage`、已完成阶段和 checkpoint；
- `*_gate.json`：该阶段实际使用的门槛；
- `*_status.json`：停止时的滑动平均指标；
- `model_flat_jump_init.pt`：需要时生成的 Flat→Jump 转换模型。

当请求区间全部通过时，`state.json` 中 `passed=true` 表示“指定的开始/结束
区间已经完成”，不一定表示九阶段全部完成。若某阶段用完最大 iteration 仍
未通过，流水线立即返回非零状态并保留 checkpoint，不会把未达标模型迁移到
更困难阶段。

## 7. 调整原则

自动门槛用于可靠切换阶段，而不是替代最终 Play 验收。建议：

- 不因成功率波动而取消 `torque_saturation`、恢复失败、真实轮端净空等安全条件；
- 障碍物阶段同时检查课程窗口和累计指标，避免某个短窗口偶然通过；
- 若确定性 Play 已合格而训练态只差少量，可按日志分布微调对应门槛；
- 若碰撞或净空不合格，应改奖励、动作时序或课程，而不是继续降低最终成功定义。
