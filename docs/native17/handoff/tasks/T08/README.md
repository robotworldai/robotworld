# T08 · 轮足机器人在起伏与台阶地形中抗扰行进

来源：[zyicome/Wheel-Legged-Lab](https://github.com/zyicome/Wheel-Legged-Lab)

固定 commit：`e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7`

原 task/config：`Wheel-Legged-Terrain-Reactive-v0`

运行依赖：作者开发版本：Isaac Sim 5.1.0 + Isaac Lab 2.3.2

接入状态：本包未实现或验证 RoboWorld_Scaffold adapter；提供原生任务接入材料。

## 资产与获取

### 开链轮足机器人 URDF

- 类型：repository
- 位置：`source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/assets/wheellegged_description/urdf/wl_dealed.urdf`
- 核查：配置/上游目录指向该位置；本轮未下载资产本体
- 定义：[源码](https://github.com/zyicome/Wheel-Legged-Lab/blob/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/assets/wheellegged.py)
- 获取/来源：[上游位置](https://github.com/zyicome/Wheel-Legged-Lab/blob/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/assets/wheellegged_description/urdf/wl_dealed.urdf)
- 配套：source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/assets/wheellegged_description/meshes/ 与 ASSET_LICENSE.md
- 注意：Isaac URDF importer 转换；两个任务共享同一资产。

### Reactive 斜坡/起伏/台阶

- 类型：procedural
- 位置：`WheelLeggedTerrainReactiveEnvCfg 中 terrain generator`
- 核查：运行时由所列源码生成，没有独立资产包
- 定义：[源码](https://github.com/zyicome/Wheel-Legged-Lab/blob/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/wheel_legged_terrain_env_cfg.py)
- 获取/来源：[上游位置](https://github.com/zyicome/Wheel-Legged-Lab/blob/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/wheel_legged_terrain_env_cfg.py)
- 注意：地形难度与随机种子是评测 setting 的一部分。

## 原生 setting

使用 Reactive 原生地形设置；actor 没有前方高度扫描。推扰间隔 5–9 s、最大平面速度 0.55；速度命令前后 ±0.6、转向 ±1.2。

## 安装参考

在 Isaac Lab Python 内 python -m pip install -e source/wheel_legged_robot；URDF 在首次构造时转换。

## 入口与控制边界

Recovery 与 Terrain-Reactive 共用资产。若使用作者 play.py，核对 --eval_pushes 是否保留了原推扰；不是新建两个独立场景。

## 动作接口

与 Recovery 相同 VMC 目标接口；程序承担动态平衡和轮腿协同，不能提供现成全身行走 policy。

## 观测条件

选 Reactive 版本：actor 无前方地形高度扫描，不要混成 Perceptive 版本。其余 proprioception 按原配置。

## 原始评估与失败

原倾角 >1 rad 或高度 <0.09 m 失败；同时评速度跟踪/位移，存活并不等于完成行进。

## 接入风险

与恢复题共用平台，不能称两个独立资产；需求区别是“限时恢复”与“通过接触变化的地形”。若最后缩题，可二选一。

## 为什么需要新反馈

台阶/斜坡接触后的姿态与轮腿受力需由即时本体反馈反映；运行中推扰进一步改变接触和速度。

## 反馈为何不能慢

双轮支撑在离地、落地和倾斜时会继续失稳，等模型下一轮可能已触发倾倒。配置的动作频率不能当成延迟实测。

## 难度结论

保留不同需求，但与其他 locomotion 题相关性高；不是靠换机器人型号增加独立覆盖。

## 已有视频/轨迹查看记录

源代码核查；未实际观看该精确配置的 rollout。

## 源码定位

- [Reactive 地形、推扰和动作](https://github.com/zyicome/Wheel-Legged-Lab/blob/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/wheel_legged_terrain_env_cfg.py#L377) · [包内文件](../../sources/zyicome/Wheel-Legged-Lab/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/wheel_legged_terrain_env_cfg.py)
- [任务注册](https://github.com/zyicome/Wheel-Legged-Lab/blob/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/__init__.py) · [包内文件](../../sources/zyicome/Wheel-Legged-Lab/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/__init__.py)
- [执行层实现](https://github.com/zyicome/Wheel-Legged-Lab/blob/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/mdp/actions.py) · [包内文件](../../sources/zyicome/Wheel-Legged-Lab/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/mdp/actions.py)
