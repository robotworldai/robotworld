# T03 · 机械臂持杯接住随机发射的球

来源：[LxRoboticsLab/ReflexBench](https://github.com/LxRoboticsLab/ReflexBench)

固定 commit：`8bb931485093c6d98f8729774ad01bf824964e16`

原 task/config：`BallCatching-Franka-IK-Abs-v0`

运行依赖：Isaac Lab → Isaac Sim；README 未锁精确版本，杯子引用 Isaac 5.1 资产，需先做版本探测

接入状态：本包未实现或验证 RoboWorld_Scaffold adapter；提供原生任务接入材料。

## 资产与获取

### Franka Panda

- 类型：external
- 位置：`isaaclab_assets.robots.franka.FRANKA_PANDA_HIGH_PD_CFG → 当前匹配 Isaac 资产根目录`
- 核查：外部资产；需由目标 Isaac 安装解析/下载，未验证远端下载
- 定义：[源码](https://github.com/LxRoboticsLab/ReflexBench/blob/8bb931485093c6d98f8729774ad01bf824964e16/source/reflexbench/reflexbench/tasks/manager_based/ball_catching/config/franka/ik_abs_env_cfg.py)
- 注意：资产路径由匹配的 Isaac Lab 版本解析，不能把高 PD 伺服误认为接球策略。

### 接球杯

- 类型：external
- 位置：`https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/Mugs/SM_Mug_A2.usd`
- 核查：外部资产；需由目标 Isaac 安装解析/下载，未验证远端下载
- 定义：[源码](https://github.com/LxRoboticsLab/ReflexBench/blob/8bb931485093c6d98f8729774ad01bf824964e16/source/reflexbench/reflexbench/tasks/manager_based/ball_catching/config/franka/joint_pos_env_cfg.py)
- 获取/来源：[上游位置](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/Mugs/SM_Mug_A2.usd)
- 注意：固定源码 scale=(0.015,0.015,0.015)，禁重力并有 virtual capture zone；须按原逻辑加载。

### 小球、桌面与发射器占位几何

- 类型：procedural
- 位置：`SphereCfg / CuboidCfg / CylinderCfg`
- 核查：运行时由所列源码生成，没有独立资产包
- 定义：[源码](https://github.com/LxRoboticsLab/ReflexBench/blob/8bb931485093c6d98f8729774ad01bf824964e16/source/reflexbench/reflexbench/tasks/manager_based/ball_catching/ball_catching_env_cfg.py)
- 获取/来源：[上游位置](https://github.com/LxRoboticsLab/ReflexBench/blob/8bb931485093c6d98f8729774ad01bf824964e16/source/reflexbench/reflexbench/tasks/manager_based/ball_catching/ball_catching_env_cfg.py)

## 原生 setting

使用原接球事件与发射分布；在发射前不向 agent 暴露内部 launch_pitch、launch_yaw、launch_timer。原 policy 观测没有这些未来发射参数。

## 安装参考

python -m pip install -e source/reflexbench；官方 planning 数据收集另需 cuRobo，不是构造任务的同义条件。

## 入口与控制边界

选择 BallCatching-Franka-IK-Abs-v0；保留运行中新发球。官方异步评估接口可参考，但 RobotWorld adapter 尚未实现。

## 动作接口

原环境同时提供关节与 EEF/IK 接口；此处选绝对 IK 版本。快速控制程序可以输出 EEF，不需要把“写代码”与“使用 EEF”对立起来。

## 观测条件

原 policy 观测含球状态及预测拦截量，不等同于 RGB-only；区分直接给预测量与 agent 从传感器自行估计。

## 原始评估与失败

原 task_phase 到达 4 判为 task_completed；已发射球低于 0.03 m 判落地。包括接住之后的阶段，不能把整题简化成抓手触碰球。

## 接入风险

杯子跟随 EEF 的运动由环境更新，并有捕获区/阻尼逻辑；不是完整自然抓握。原观测含预测拦截量，状态版可能明显简单，不能冒称纯视觉难题。

## 为什么需要新反馈

发射器在回合开始后才发球。发射延迟为 0.3–0.8 秒，俯仰角为 50–65°，偏航偏移为 −15–15°。必须观察实际发射后的球位置与运动方向，才能确定接球位置。

## 反馈为何不能慢

来球在重力作用下经过接球区域后落地；程序不能让球停在空中等待下一次模型回答。速度配置为 3.4 m/s，属于有限飞行窗口的拦截。

## 难度结论

有限飞行时间提供强时限证据；接球程序可能只需快速估计一次再发动作段，不要求每个物理步重新规划。

## 已有视频/轨迹查看记录

实际查看 ReflexVLA 官方真实接球对比视频约 57.62 / 134.91 s 的片段；该真机示例与所选仿真捕获机制不同。

## 源码定位

- [随机发射、跟踪与捕获事件](https://github.com/LxRoboticsLab/ReflexBench/blob/8bb931485093c6d98f8729774ad01bf824964e16/source/reflexbench/reflexbench/tasks/manager_based/ball_catching/mdp/events.py#L125) · [包内文件](../../sources/LxRoboticsLab/ReflexBench/source/reflexbench/reflexbench/tasks/manager_based/ball_catching/mdp/events.py)
- [观测与事件绑定](https://github.com/LxRoboticsLab/ReflexBench/blob/8bb931485093c6d98f8729774ad01bf824964e16/source/reflexbench/reflexbench/tasks/manager_based/ball_catching/ball_catching_env_cfg.py#L88) · [包内文件](../../sources/LxRoboticsLab/ReflexBench/source/reflexbench/reflexbench/tasks/manager_based/ball_catching/ball_catching_env_cfg.py)
- [落地与完成判据](https://github.com/LxRoboticsLab/ReflexBench/blob/8bb931485093c6d98f8729774ad01bf824964e16/source/reflexbench/reflexbench/tasks/manager_based/ball_catching/mdp/terminations.py#L39) · [包内文件](../../sources/LxRoboticsLab/ReflexBench/source/reflexbench/reflexbench/tasks/manager_based/ball_catching/mdp/terminations.py)
- [官方接球演示](https://reflexvla.github.io/)
