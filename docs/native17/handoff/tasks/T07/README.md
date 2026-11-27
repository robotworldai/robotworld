# T07 · 轮足机器人从近跌倒状态恢复直立

来源：[zyicome/Wheel-Legged-Lab](https://github.com/zyicome/Wheel-Legged-Lab)

固定 commit：`e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7`

原 task/config：`Wheel-Legged-Recovery-Flat-v0`

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

### 恢复任务平面

- 类型：procedural
- 位置：`WheelLeggedRecoveryFlatEnvCfg`
- 核查：运行时由所列源码生成，没有独立资产包
- 定义：[源码](https://github.com/zyicome/Wheel-Legged-Lab/blob/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/wheel_legged_terrain_env_cfg.py)
- 获取/来源：[上游位置](https://github.com/zyicome/Wheel-Legged-Lab/blob/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/wheel_legged_terrain_env_cfg.py)

## 原生 setting

reset 随机倾角/角速度；运行中每 3–6 s 推扰、最大平面速度 0.9 m/s。恢复窗口 2.5 s，启动宽限 0.25 s。

## 安装参考

在 Isaac Lab Python 内 python -m pip install -e source/wheel_legged_robot；URDF 在首次构造时转换。

## 入口与控制边界

Recovery 与 Terrain-Reactive 共用资产。若使用作者 play.py，核对 --eval_pushes 是否保留了原推扰；不是新建两个独立场景。

## 动作接口

6 维腿角/腿长/轮速目标，通过已有 VMC 与伺服执行。已读 action 实现：该层跟踪腿/轮目标，不替 agent 用躯干姿态完成平衡。

## 观测条件

保留原生观测定义；源码配置中可能包含 simulator state。原生 state 条件与 RGB/RGB-D 条件需分开记录，本包不擅自宣称纯视觉。

## 原始评估与失败

原恢复判据：倾角 <0.25 rad、高度 >0.14 m，保持 0.30 s；超恢复期限或倒地失败。

## 接入风险

是近跌倒恢复，不是从完全侧躺地面起身。默认 play 可能移除推扰，作者 --eval_pushes 用于保留推扰。

## 为什么需要新反馈

实际倾角、角速度、轮速和接触决定该向哪边驱动车轮及如何调节腿长；随机推扰发生后不能沿原动作串继续。

## 反馈为何不能慢

恢复必须在 2.5 s 内完成并连续稳定 0.30 s；倾角超过 1 rad 或高度低于 0.085 m 提前失败。2.5 s 不是每次可以等 2.5 s 再控制。

## 难度结论

原设置有恢复期限与连续稳定判据，比“保持不倒”更可解释；暂未测简单 LQR/PD 能达到多高。

## 已有视频/轨迹查看记录

查到 README 的行走/跳跃 demo；未观看精确 Recovery 设置的完整成功失败轨迹，因此不拿跳跃视频当它的验证。

## 源码定位

- [恢复判据](https://github.com/zyicome/Wheel-Legged-Lab/blob/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/wheel_legged_terrain_env_cfg.py#L197) · [包内文件](../../sources/zyicome/Wheel-Legged-Lab/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/wheel_legged_terrain_env_cfg.py)
- [恢复环境配置](https://github.com/zyicome/Wheel-Legged-Lab/blob/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/wheel_legged_terrain_env_cfg.py#L297) · [包内文件](../../sources/zyicome/Wheel-Legged-Lab/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/wheel_legged_terrain_env_cfg.py)
- [底层 VMC/动作职责](https://github.com/zyicome/Wheel-Legged-Lab/blob/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/mdp/actions.py#L199) · [包内文件](../../sources/zyicome/Wheel-Legged-Lab/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/mdp/actions.py)
