# 本地 Codex 全项目冒烟评测交接

交接记录生成时间：2026-09-29T14:33:52.933172+08:00。批次 `smoke-native-one-per-bench-20260929-01` 已恢复，当前状态请以持续更新的 campaign.json 为准。

## 结果入口

- [持续更新的结果与视频列表](outputs/_smoke/smoke-native-one-per-bench-20260929-01/README.md)
- [结构化状态、逐题结果与运行路径](outputs/_smoke/smoke-native-one-per-bench-20260929-01/campaign.json)
- [后台驱动日志](outputs/_smoke/smoke-native-one-per-bench-20260929-01/resume-driver-02.log)
- [驱动 PID、启动命令与时间](outputs/_smoke/smoke-native-one-per-bench-20260929-01/resume-driver-02.json)
- [评测脚本、目录、统计与 resume 说明](scripts/ROLLOUTS.md)

## 本次范围与运行方式

20 个 benchmark，每个项目选一题、一次 rollout。这是完整链路冒烟，不是全部 85 个入口的多次成功率评测。保留题目登记的步数上限和原生提前终止，未缩短为统一 32/300/600 步。模型为 gpt-6-astra，使用 World/codex 源码构建的 app-server 和本地环境；没有用系统安装的 Codex 替代，也没有由本交互代理代做动作。认证来自本地用户的 Codex 配置目录，不应随交接公开认证文件。

当前后台驱动 PID 为 3676290，独立进程组，关闭本轮对话不需要重新启动。串行运行以避免抢占单卡显存；完成或发生可记录的运行错误后自动进入下一题。已完成的 RoboCasa、RoboDojo 跳过，BEHAVIOR 排最后重跑。

RoboCasa / CoffeeSetupMug 的已有结果 success=true；RoboDojo / deposit_coin 为 success=false，两者冒烟链路检查均通过。其他项目尚不能据启动状态宣称成功。模型/环境异常、原生失败、原生 reward-only 无二值 SR 分别记录。视频按模拟控制步录制，不按 LLM history 抽帧；未正常完成的 MP4 可能尚未封装完。

## 本批次顺序与预算

| 项目 | 题目 | 登记控制步上限 |
|---|---|---:|
| robocasa | CoffeeSetupMug | 600 |
| robodojo | deposit_coin | 300 |
| robolab | RubiksCubeLeftOfBowlTask | 450 |
| humanoid_soccer | play-soccer | 300 |
| ai_cps | 22 | 300 |
| wheeledlab | mushr-drift | 250 |
| bench2dex | 41 | 681 |
| steadytray | tray_balancing_walk（端托盘行走，同时抵抗物体与身体推扰） | 1000 |
| ttrl | humanoid_table_tennis_return（人形机器人连续接回随机乒乓来球） | 540 |
| reflexbench | cup_ball_catching（机械臂持杯接住随机发射的球） | 100 |
| volleybots | drone_volleyball_solo_juggle（无人机单机颠球） | 800 |
| wheel_legged | wheel_legged_upright_recovery（轮足机器人从近跌倒状态恢复直立） | 2000 |
| wheeled_quadruped | rear_wheel_upright_balance（四足机器人以后两轮直立并抵抗随机推扰） | 1000 |
| go2_push | quadruped_push_recovery（四足机器人行走时抵抗随机外力脉冲） | 1000 |
| robot_lab | a1_front_leg_handstand（A1 抬起后腿、以前腿支撑并抗扰） | 500 |
| digit | digit_walk_hand_tracking（Digit 行走时同时跟踪双手目标） | 700 |
| omniisaacgymenvs | anymal_rough_terrain（四足崎岖地形行走并抵抗推扰） | 1000 |
| flamingo | wheel_legged_jump_balance（双轮足机器人按指令跳起、落地并继续平衡） | 1000 |
| omnidrones | drone_payload_hover（无人机吊载悬停并抑制受扰摆动） | 500 |
| behavior_1k | clean_up_your_desk | 32126 |

预算来源在 budget-provenance.json 和各 run.json 中。特别是 T02 的 540 是接入层保守预算，不是上游统一官方回合时长；Isaac Sim 6 实验兼容配置不宣称物理等价。BEHAVIOR 32126 来自该任务官方 human length × 1.5，并非 32126 次模型推理。

## BEHAVIOR 本次重跑的变化

- 上次书桌整理中断于已记录 2213 控制步，旧视频和轨迹留在原 run，不能计入完整任务成功率。
- 新尝试会使用 retry-01 等新目录，保留旧尝试；不恢复场景、Codex 上下文或程序 memory。
- 新增 coding_control：每个新观测回调一次，单段最多 600 步，动作仍经过原 move_robot 映射，原生整题判定与预算不变。
- 新增 BEHAVIOR 专用小幅动作提示词：3 cm / 5°，近接触 5 mm / 2°；程序按 dt 生成渐进参考目标并检查误差。这是提示词约束，不是执行端硬限速，效果以新尝试为准。
- 没有改写上游环境源码、物理或成功判据，也没有更改旧 prompt 与视频。

## 查看与恢复

```bash
cd /path/to/robotworld
tail -f outputs/_smoke/smoke-native-one-per-bench-20260929-01/resume-driver-02.log
cat outputs/_smoke/smoke-native-one-per-bench-20260929-01/campaign.json
```

如果驱动已经退出且还有中断/运行错误项，确认旧环境进程与容器已停止后，使用相同批次续跑；不要在当前驱动仍在运行时另开一份。

```bash
bash scripts/smoke_all.sh run --resume --batch smoke-native-one-per-bench-20260929-01 \
  --code-control on --model gpt-6-astra --codex-home "$HOME/.codex"
```

每个逻辑 rollout 只把最新有效尝试计入统计，旧 attempt 保留在 previous_attempts。已完成的原生失败也会跳过，不自动刷成功；独立重复实验需新建 batch。

## 保存目录

```text
outputs/<benchmark>/<task>/run-<batch>-0001[-retry-01]/
  run.json             # 状态、配置、原生结果来源、历史尝试
  videos/              # 指向 artifacts 原始 MP4 的索引
  events/              # 完整 Codex/工具/环境日志及 no-images 版本
  programs/            # 程序控制代码（使用时）
  agent-workspace/     # Codex 写下的程序和笔记（适用时）
  artifacts/           # 原启动器完整记录
```

该表只登记实际产生的文件；环境初始化受阻或录像尚未关闭时不会伪称有完整视频。RoboDojo 禁用 shell，工作区检查不适用。

T04 / aerial_balance 已按用户要求从当前批次和默认入口移除；历史报错、上游源码和资产仅保留归档。当前有效冒烟题数为 20。

任务名称与旧编号对照见 [TASK_NAMES.md](docs/TASK_NAMES.md)，summary 字段说明见 [SUMMARY_FORMAT.md](docs/SUMMARY_FORMAT.md)。现有 outputs 目录已使用具体任务名，旧编号路径为兼容软链接。

统一 0–100 综合分的设计和当前各项目指标核查见 [UNIFIED_SCORE.md](docs/UNIFIED_SCORE.md)。尚未校准 reward 端点，不发布未经验证的总分。
