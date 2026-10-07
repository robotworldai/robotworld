# World benchmark 与题目清单

2026-09-29增补：Bench2Dex 新增 RobotWorld 41–49 共9道题，见[独立清单及运行状态](../third_party/benchmarks/bench2dex/README.md)。下方总数与其他项目状态仍是2026-09-28快照，未计入本次增补；不要将其当作当前全项目总数。

统计日期：2026-09-28。依据当前 World 注册清单和验证记录，不代表上游全部任务均可用。

**20 个来源项目、50 个 case/模式入口；按本文件口径去重为 46 个任务或独立自定义协议条目。46 包含 T04/T05/T15 三个明确阻塞项，不是“46 题全部跑通”。**

去重口径：RoboCasa ID28 与 ID10 合并；足球 baseline/hybrid/direct 是同一场景的控制模式，合并统计。AI-CPS ID34 与 ID24 共用环境，但判据不同，因此按两个协议统计。T06 不重复计入 WheeledLab；T12 的现有 MuJoCo 版本不能当作交接包原 Isaac 任务已跑通。题号 ID11 与 T11 等属于两套编号体系，不要混用。

## 项目汇总

| 项目 | 去重任务/协议条目数 | 当前题目 |
|---|---:|---|
| RoboDojo (`robodojo`) | 2 | conveyor：传送带匹配抓取；coin：硬币投放 |
| BEHAVIOR-1K (`behavior_1k`) | 2 | groceries：跨房间杂货搬运与冰箱收尾；desk：儿童房书桌整理 |
| RoboCasa (`robocasa`) | 5 | 8：清洁厨房台面并确认完成；9：餐具分类、清洗与归位；10：咖啡杯对准出液口并保持终态；14：抽屉未完全关闭时重新推入；17：导航到设施后根据设施状态选择操作 |
| NVlabs RoboLab (`robolab`) | 6 | cube-left：魔方放到碗左侧；cube-front：魔方放到碗前方；hammers：锤子整理；other-tools：非锤工具整理；cans：罐头装箱；mugs：两个杯子上架 |
| HumanoidSoccer (`humanoid_soccer`) | 1 | play-soccer：固定球场：代码控制 G1 行走与踢球 |
| AI-CPS Robotics Benchmark (`ai_cps`) | 4 | 22：运动小球接取；23：托盘小球平衡；24：Peg-in-hole 插入；34：真实接触异常后的取消与降级控制 |
| WheeledLab (`wheeledlab`) | 11 | mushr-drift：小车在随机扰动下连续漂移过弯（MuSHR）；f1tenth-drift：F1TENTH 漂移；elevation：地形导航；visual：视觉导航；rw-courtyard：园区配送：绕箱、缓坡、定向停车；rw-hairpins：高架山路：窄桥与连续回头弯；rw-gate-dock：动态闸门：停车让行、限速窄口与精准停靠；rw-drift-switch：变摩擦赛道：受扰连续漂移与闭环恢复；rw-twin-beam：精密双窄梁桥：对准轮迹、上桥与定向停车；rw-reverse-bay：狭窄倒车入库：夹车通道、倒入与厘米级停正；rw-parallel-park：夹车侧方停车：多次进退与平行精确停靠 |
| SteadyTray (`steadytray`) | 1 | T01：端托盘行走，同时抵抗物体与身体推扰 |
| TTRL-ICRA2026 (`ttrl`) | 1 | T02：人形机器人连续接回随机乒乓来球 |
| ReflexBench (`reflexbench`) | 1 | T03：机械臂持杯接住随机发射的球 |
| aerial_balance_bench (`aerial_balance`) | 1 | T04：绳牵横梁上的小球抗扰定位 |
| VolleyBots (`volleybots`) | 1 | T05：无人机一对一排球对抗 |
| Wheel-Legged-Lab (`wheel_legged`) | 2 | T07：轮足机器人从近跌倒状态恢复直立；T08：轮足机器人在起伏与台阶地形中抗扰行进 |
| wheeled_quadruped_robot (`wheeled_quadruped`) | 1 | T09：四足机器人以后两轮直立并抵抗随机推扰 |
| isaaclab-go2-locomotion (`go2_push`) | 1 | T10：四足机器人行走时抵抗随机外力脉冲 |
| AgileX robot_lab (`robot_lab`) | 1 | T11：A1 抬起后腿、以前腿支撑并抗扰 |
| IsaacLab / Digit (`digit`) | 1 | T13：Digit 行走时同时跟踪双手目标 |
| OmniIsaacGymEnvs (`omniisaacgymenvs`) | 1 | T14：四足崎岖地形行走并抵抗推扰 |
| Isaac-RL-Two-wheel-Legged-Bot (`flamingo`) | 1 | T15：双轮足机器人按指令跳起、落地并继续平衡 |
| OmniDrones (`omnidrones`) | 2 | T16：无人机吊载悬停并抑制受扰摆动；T17：无人机托举倒立摆并跟踪轨迹 |

## 接入状态与限制

- RoboDojo、BEHAVIOR-1K、RoboCasa、NVlabs RoboLab、HumanoidSoccer、AI-CPS、WheeledLab 已有接入和历史运行记录；这里不把有入口当作每个条目均完成整回合验证，也不把执行成功当作任务成功。
- Native17 新增 15 题中，12 题有历史模型回合；T04/T05/T15 仍阻塞。T11 历史回合使用修复前摩擦参数，修复后已做环境回归与旧动作重放，尚需新模型闭环，不能把旧成绩标作当前修复版本成绩。
- T04：缺作者四个原 payload；T05：旧版 GPU 软件栈不支持本机且缺目标 1v1 完整对手；T15：原配置删除 height_scanner，但 critic 仍引用。
- 大量 Isaac 任务当前使用 6.0.1 实验兼容栈，不代表官方物理数值等价。
- RoboCasa 六组 600 步历史诊断：CloseDrawer、NavigateKitchen 成功；其他四组未成功。它们是单回合诊断，不是统计成功率。ID8 当前统一入口需显式 --steps；ID10/28 是同一任务的不同种子/分析视角。
- NVlabs RoboLab 的实际清单是六个原任务；曾提出的满容器选择、强制滑落、传感器缺失等九项需求，不等于九个已经独立实现的官方任务。
- HumanoidSoccer 当前是 MuJoCo sim2sim，固定 play-soccer 含自定义控制诊断与平衡辅助；不是 Native17 T12 的原 Isaac 任务协议。
- AI-CPS ID34 是自定义真实接触恢复协议；24/34 在当前 PhysX 有碰撞凸包回退限制。
- WheeledLab 11 条目 = 4 个原环境 + 7 个 RobotWorld 自定义驾驶协议；自定义题不能混称官方 benchmark 任务。

## 完整入口（含别名和模式）

| 项目 | ID | native task / protocol | 类型 | 备注 |
|---|---|---|---|---|
| robodojo | conveyor | `match_and_pick_from_conveyor` | native_task |  |
| robodojo | coin | `deposit_coin` | native_task |  |
| behavior_1k | groceries | `carrying_in_groceries` | native_task |  |
| behavior_1k | desk | `clean_up_your_desk` | native_task |  |
| robocasa | 8 | `CountertopCleanup` | native_task | 需要显式指定步数；官方 horizon 未核实 |
| robocasa | 9 | `SortingCleanup` | native_task |  |
| robocasa | 10 | `CoffeeSetupMug` | native_task |  |
| robocasa | 14 | `CloseDrawer` | native_task |  |
| robocasa | 17 | `NavigateKitchen` | native_task |  |
| robocasa | 28 | `CoffeeSetupMug` | alias | 与 ID10 同一官方任务 |
| robolab | cube-left | `RubiksCubeLeftOfBowlTask` | native_task |  |
| robolab | cube-front | `RubiksCubeInFrontOfBowlTask` | native_task |  |
| robolab | hammers | `ToolOrganizationBothTask` | native_task |  |
| robolab | other-tools | `NonHammerToolsInRightBinTask` | native_task |  |
| robolab | cans | `FoodPacking2CansTask` | native_task |  |
| robolab | mugs | `PutTwoMugsOnShelfTask` | native_task |  |
| humanoid_soccer | play-soccer | `official_mujoco_soccer` | custom_protocol |  |
| humanoid_soccer | baseline | `official_mujoco_soccer` | control_variant |  |
| humanoid_soccer | hybrid | `official_mujoco_soccer` | control_variant |  |
| humanoid_soccer | direct | `official_mujoco_soccer` | control_variant |  |
| ai_cps | 22 | `FrankaBallCatching` | native_task |  |
| ai_cps | 23 | `FrankaBallBalancing` | native_task |  |
| ai_cps | 24 | `FrankaPegInHole` | native_task |  |
| ai_cps | 34 | `FrankaPegInHole` | custom_protocol |  |
| wheeledlab | mushr-drift | `Isaac-MushrDriftRL-v0` | native_task |  |
| wheeledlab | f1tenth-drift | `Isaac-F1TenthDriftRL-v0` | native_task |  |
| wheeledlab | elevation | `Isaac-MushrElevationRL-v0` | native_task |  |
| wheeledlab | visual | `Isaac-MushrVisualRL-v0` | native_task |  |
| wheeledlab | rw-courtyard | `RobotWorld-rw-courtyard-v1` | custom_protocol |  |
| wheeledlab | rw-hairpins | `RobotWorld-rw-hairpins-v1` | custom_protocol |  |
| wheeledlab | rw-gate-dock | `RobotWorld-rw-gate-dock-v1` | custom_protocol |  |
| wheeledlab | rw-drift-switch | `RobotWorld-rw-drift-switch-v1` | custom_protocol |  |
| wheeledlab | rw-twin-beam | `RobotWorld-rw-twin-beam-v1` | custom_protocol |  |
| wheeledlab | rw-reverse-bay | `RobotWorld-rw-reverse-bay-v1` | custom_protocol |  |
| wheeledlab | rw-parallel-park | `RobotWorld-rw-parallel-park-v1` | custom_protocol |  |
| steadytray | T01 | `G1-Steady-Object` | native_task | 有实验兼容模型回合；不等于任务成功 |
| ttrl | T02 | `t1_tt_eval` | native_task | 有实验兼容模型回合；不等于任务成功 |
| reflexbench | T03 | `BallCatching-Franka-IK-Abs-v0` | native_task | 有实验兼容模型回合；不等于任务成功 |
| aerial_balance | T04 | `template_eval_disturbance.yaml` | native_task | 缺作者四个 payload 资产 |
| volleybots | T05 | `Volleyball1v1` | native_task | 旧 GPU 栈不支持当前硬件，且缺目标1v1完整对手 |
| wheel_legged | T07 | `Wheel-Legged-Recovery-Flat-v0` | native_task | 有实验兼容模型回合；不等于任务成功 |
| wheel_legged | T08 | `Wheel-Legged-Terrain-Reactive-v0` | native_task | 有实验兼容模型回合；不等于任务成功 |
| wheeled_quadruped | T09 | `Wheeled-Quadruped-Balance-v0` | native_task | 有实验兼容模型回合；不等于任务成功 |
| go2_push | T10 | `Isaac-Velocity-Flat-Unitree-Go2-PushRecovery-v0` | native_task | 有实验兼容模型回合；不等于任务成功 |
| robot_lab | T11 | `RobotLab-Isaac-Velocity-Flat-HandStand-Unitree-A1-v0` | native_task | 摩擦修复已回归，旧模型动作已重放；修复后新闭环尚未测 |
| digit | T13 | `Isaac-Tracking-LocoManip-Digit-v0` | native_task | 有实验兼容模型回合；不等于任务成功 |
| omniisaacgymenvs | T14 | `AnymalTerrain` | native_task | 有实验兼容模型回合；不等于任务成功 |
| flamingo | T15 | `Isaac-TrackJUMP-Flat-Flamingo-v1-ppo` | native_task | 原配置移除 height_scanner 但 critic 仍引用 |
| omnidrones | T16 | `Payload/PayloadHover` | native_task | 有实验兼容模型回合；不等于任务成功 |
| omnidrones | T17 | `InvPendulum/InvPendulumTrack` | native_task | 有实验兼容模型回合；不等于任务成功 |

## 数据来源与运行入口

- [常规七项目清单](../environment/evaluation/suites.json)
- [Native17 清单](../environment/evaluation/native17.json)
- [Native17 验证记录](native17/VALIDATION.md)
- [每 benchmark 的运行脚本](../scripts/eval/README.md)
- [T11 最新修复报告](../var/runs/docker/robot_lab/physics-friction-fix-report.md)
- [RoboCasa 六组诊断](../var/runs/docker/robocasa/six-cases-600steps-01/README.md)
- [可程序读取的同版统计 JSON](BENCHMARK_INVENTORY.json)

在 World 根目录运行 `bash scripts/eval/all.sh list` 查看常规清单；`bash scripts/eval/native17.sh list` 查看新增题目。WheeledLab 自定义与精密驾驶另有 `wheeledlab_custom.sh list`、`wheeledlab_precision.sh list`。
