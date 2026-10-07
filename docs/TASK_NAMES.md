# 任务名称与历史编号

脚本参数、新运行目录和结果汇总使用下列具体任务名。旧编号仍可作为命令行别名使用；上游环境的内部注册 ID、原始轨迹和历史命令保持不变。历史 outputs 编号目录是指向新名称目录的相对软链接，不重复保存视频。

| benchmark | 任务名（CLI / outputs） | 中文任务 | 历史别名 |
|---|---|---|---|
| steadytray | `tray_balancing_walk` | 端托盘行走，同时抵抗物体与身体推扰 | `T01` |
| ttrl | `humanoid_table_tennis_return` | 人形机器人连续接回随机乒乓来球 | `T02` |
| reflexbench | `cup_ball_catching` | 机械臂持杯接住随机发射的球 | `T03` |
| volleybots | `drone_volleyball_1v1` | 无人机一对一排球对抗 | `T05` |
| wheeledlab | `mushr-drift` | 小车在随机扰动下连续漂移过弯 | `T06` |
| wheel_legged | `wheel_legged_upright_recovery` | 轮足机器人从近跌倒状态恢复直立 | `T07` |
| wheel_legged | `wheel_legged_rough_terrain` | 轮足机器人在起伏与台阶地形中抗扰行进 | `T08` |
| wheeled_quadruped | `rear_wheel_upright_balance` | 四足机器人以后两轮直立并抵抗随机推扰 | `T09` |
| go2_push | `quadruped_push_recovery` | 四足机器人行走时抵抗随机外力脉冲 | `T10` |
| robot_lab | `a1_front_leg_handstand` | A1 抬起后腿、以前腿支撑并抗扰 | `T11` |
| humanoid_soccer | `play-soccer` | 人形机器人抗扰平衡并踢移动足球 | `T12` |
| digit | `digit_walk_hand_tracking` | Digit 行走时同时跟踪双手目标 | `T13` |
| omniisaacgymenvs | `anymal_rough_terrain` | 四足崎岖地形行走并抵抗推扰 | `T14` |
| flamingo | `wheel_legged_jump_balance` | 双轮足机器人按指令跳起、落地并继续平衡 | `T15` |
| omnidrones | `drone_payload_hover` | 无人机吊载悬停并抑制受扰摆动 | `T16` |
| omnidrones | `drone_inverted_pendulum_tracking` | 无人机托举倒立摆并跟踪轨迹 | `T17` |
| volleybots | `drone_volleyball_solo_juggle` | 无人机单机颠球 | `T05-single` |

T04 已移出活动题集，不提供新任务名。小车和足球沿用已有入口；足球的 MuJoCo 集成并非原始 Isaac 任务的等价评测。

```bash
bash scripts/run_omnidrones.sh list
bash scripts/run_omnidrones.sh run --tasks drone_payload_hover \
  --task-steps drone_payload_hover=native --rollouts 3 \
  --codex-home /path/to/codex-home --batch payload-modelA
```

需要迁移其他旧 outputs 时，可先预览 `python scripts/migrate_task_names.py --output-root PATH`，确认后添加 `--apply`。脚本拒绝迁移标记为 running 的任务以及合并冲突目录，保留旧路径软链接；不修改原始模拟器结果和 Codex 轨迹。
