# T15 · 双轮足机器人按指令跳起、落地并继续平衡

来源：[jaykorea/Isaac-RL-Two-wheel-Legged-Bot](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot)

固定 commit：`d922cce9e07a8877c37e12b2cb39dfdcf34b9d40`

原 task/config：`Isaac-TrackJUMP-Flat-Flamingo-v1-ppo`

运行依赖：Isaac Sim 4.5 + Isaac Lab 2.0.0 + Python 3.10（README）

接入状态：本包未实现或验证 RoboWorld_Scaffold adapter；提供原生任务接入材料。

## 资产与获取

### Flamingo rev01_5_2 资产 ZIP

- 类型：repository
- 位置：`lab/flamingo/assets/data/Robots/Flamingo/flamingo_rev01_5_2/flamingo_rev01_5_2_merge_joints.zip`
- 核查：配置/上游目录指向该位置；本轮未下载资产本体
- 定义：[源码](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/d922cce9e07a8877c37e12b2cb39dfdcf34b9d40/lab/flamingo/assets/flamingo/flamingo_rev01_5_2.py)
- 获取/来源：[上游位置](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/d922cce9e07a8877c37e12b2cb39dfdcf34b9d40/lab/flamingo/assets/data/Robots/Flamingo/flamingo_rev01_5_2/flamingo_rev01_5_2_merge_joints.zip)
- 配套：lab/flamingo/assets/data/ActuatorNets/Flamingo/kan/wheel/symbolic_formula.txt（按所选 actuator 配置检查）
- 注意：按 README 在原目录解压，目标为同名 .usd；同时保留 USD 引用。

## 原生 setting

原跳跃事件命令每 3–5 s 更新，事件持续 1.2 s；20 s episode，13–15 s 身体速度推扰。物理 200 Hz、动作 50 Hz。

## 安装参考

按 README 安装 lab.flamingo，手动解压指定资产 ZIP。不要按 README 示例下载另一个 rev 的机器人。

## 入口与控制边界

Isaac-TrackJUMP-Flat-Flamingo-v1-ppo；非 Play；TrackJump 使用 rev01_5_2，而不是 README 安装示例中的 rev01_4_1。

## 动作接口

髋/腿关节位置与轮速度目标；下层是执行器控制，程序承担跳跃和姿态协调。

## 观测条件

保留原生观测定义；源码配置中可能包含 simulator state。原生 state 条件与 RGB/RGB-D 条件需分开记录，本包不擅自宣称纯视觉。

## 原始评估与失败

原奖励包含事件内上升速度和蹬地，身体/髋肩腿触地等失败；没有现成完整“跳跃+稳定落地”二值 SR。

## 接入风险

Play 把推扰幅度全置 0，即使 interval 仍存在。必须用非 Play 原配置；不能把已知跳跃时序当作必须闭环的唯一理由。

## 为什么需要新反馈

实际起跳速度、空中姿态和落地时刻决定伸缩腿与轮速的切换；推扰和接触差异不能由一个固定跳跃时序覆盖。

## 反馈为何不能慢

飞行与落地不等待模型，错过接地控制会倒地。不是仅按节拍伸缩腿的动画。

## 难度结论

保留起跳/飞行/落地转换；与轮足恢复题相关，正式集应报告机制家族而不只报任务数量。

## 已有视频/轨迹查看记录

检查官方 README 动图入口与配置，未实际观看精确 TrackJump 的完整 rollout。

## 源码定位

- [跳跃目标与运行推扰](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/d922cce9e07a8877c37e12b2cb39dfdcf34b9d40/lab/flamingo/tasks/manager_based/locomotion/velocity/flamingo_env/flat_env/track_jump/flat_env_track_jump_cfg.py#L190) · [包内文件](../../sources/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/lab/flamingo/tasks/manager_based/locomotion/velocity/flamingo_env/flat_env/track_jump/flat_env_track_jump_cfg.py)
- [Play 推扰为零](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/d922cce9e07a8877c37e12b2cb39dfdcf34b9d40/lab/flamingo/tasks/manager_based/locomotion/velocity/flamingo_env/flat_env/track_jump/flat_env_track_jump_cfg.py#L253) · [包内文件](../../sources/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/lab/flamingo/tasks/manager_based/locomotion/velocity/flamingo_env/flat_env/track_jump/flat_env_track_jump_cfg.py)
- [动作与时钟](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/d922cce9e07a8877c37e12b2cb39dfdcf34b9d40/lab/flamingo/tasks/manager_based/locomotion/velocity/flamingo_env/velocity_env_cfg.py) · [包内文件](../../sources/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/lab/flamingo/tasks/manager_based/locomotion/velocity/flamingo_env/velocity_env_cfg.py)
- [任务注册](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/d922cce9e07a8877c37e12b2cb39dfdcf34b9d40/lab/flamingo/tasks/manager_based/locomotion/velocity/flamingo_env/__init__.py#L102) · [包内文件](../../sources/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/lab/flamingo/tasks/manager_based/locomotion/velocity/flamingo_env/__init__.py)
