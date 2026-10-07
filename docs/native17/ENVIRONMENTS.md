# 环境与入口

版本来自固定源码/README，均未在本机 GPU 上安装验证。不要把不同版本的任务默认迁移到同一个最新 Isaac 环境。

## T01 · AllenHuangGit/SteadyTray

- 固定版本：`ae781f9f43f9505118c2c434733c57ed1c09dbc3`
- 依赖：Isaac Sim 4.5 + 作者 IsaacLab_SteadyTray 分支；不可直接替换成官方同名版本
- 安装参考：按 README 的作者 fork + Docker 流程安装，再安装 SteadyTray 扩展；依赖 fork 的 termination manager 已附固定版本证据。
- 入口提醒：G1-Steady-Object 是原 Stage 3 环境；作者训练 runner 使用残差训练。测试 agent 时，必须记录是接原完整关节动作还是接作者残差 runner，不能混为同一个 setting。
- 上游：[README](https://github.com/AllenHuangGit/SteadyTray/blob/ae781f9f43f9505118c2c434733c57ed1c09dbc3/README.md)

## T02 · purdue-tracelab/TTRL-ICRA2026

- 固定版本：`fdb192f8fff9f5ca54c4ec4a8441fda5174ca078`
- 依赖：Isaac Sim 4.5.0 + Isaac Lab 2.1.0；作者提示升级 Sim 5.0+ 会改变成绩
- 安装参考：按 README 安装 legged_lab 与作者 runner；先核对资产，再用 t1_tt_eval。
- 入口提醒：t1_tt_eval 为作者任务名；直接从注册表构造。--predictor 是作者额外预测器，不应悄悄赠送给 agent；若使用需标注观测条件。
- 上游：[README](https://github.com/purdue-tracelab/TTRL-ICRA2026/blob/fdb192f8fff9f5ca54c4ec4a8441fda5174ca078/README.md)

## T03 · LxRoboticsLab/ReflexBench

- 固定版本：`8bb931485093c6d98f8729774ad01bf824964e16`
- 依赖：Isaac Lab → Isaac Sim；README 未锁精确版本，杯子引用 Isaac 5.1 资产，需先做版本探测
- 安装参考：python -m pip install -e source/reflexbench；官方 planning 数据收集另需 cuRobo，不是构造任务的同义条件。
- 入口提醒：选择 BallCatching-Franka-IK-Abs-v0；保留运行中新发球。官方异步评估接口可参考，但 RobotWorld adapter 尚未实现。
- 上游：[README](https://github.com/LxRoboticsLab/ReflexBench/blob/8bb931485093c6d98f8729774ad01bf824964e16/README.md)

## T04 · Wenminggong/aerial_balance_bench

- 固定版本：`d5bf6eae9f955bc03609d9adaedcc9885b31d6b9`
- 依赖：Isaac Sim 4.2.0 + Isaac Lab 1.4.0 + Python 3.10（作者 tested stack）
- 安装参考：按 README 安装 conda_env.yml；在配置好的 Isaac Lab Python 中运行。先用 zero_action_policy_eval.py 核查接口。
- 入口提醒：不是 Gym ID：使用 environments/configs/template_eval_disturbance.yaml。默认 delay-free 配置不是本清单选择的题。
- 上游：[README](https://github.com/Wenminggong/aerial_balance_bench/blob/d5bf6eae9f955bc03609d9adaedcc9885b31d6b9/README.md)

## T05 · thu-uav/VolleyBots

- 固定版本：`10e3701e480b518041c8be6a40efee7bdcb69283`
- 依赖：Isaac Sim 2023.1.0-hotfix.1 + 作者 Orbit/依赖；独立旧版本容器
- 安装参考：README 提供 jimmyzhangruize/isaac-sim:2023.1.0-hotfix.1 镜像；需按 README mount 克隆目录。镜像未在本机拉取。
- 入口提醒：Volleyball1v1；锁定对手与 checkpoint。checkpoint 目录有作者基线，先核对所用文件，不把分层 Serve/Attack 技能接为 agent 免费小脑。
- 上游：[README](https://github.com/thu-uav/VolleyBots/blob/10e3701e480b518041c8be6a40efee7bdcb69283/README.md)

## T06 · UWRobotLearning/WheeledLab

- 固定版本：`a5017ce0696032805190448b2addb8077e458387`
- 依赖：Isaac Sim 4.5.0 + Isaac Lab 2.0.2 + Python 3.10；README 要求 numpy<2
- 安装参考：从 source 目录 pip install -e wheeledlab / wheeledlab_tasks / wheeledlab_assets / wheeledlab_rl，四个扩展分别安装。
- 入口提醒：Isaac-MushrDriftRL-v0 / RSS_DRIFT_CONFIG；不要使用取消 reward/termination 的 Play 配置计分。
- 上游：[README](https://github.com/UWRobotLearning/WheeledLab/blob/a5017ce0696032805190448b2addb8077e458387/README.md)

## T07, T08 · zyicome/Wheel-Legged-Lab

- 固定版本：`e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7`
- 依赖：作者开发版本：Isaac Sim 5.1.0 + Isaac Lab 2.3.2
- 安装参考：在 Isaac Lab Python 内 python -m pip install -e source/wheel_legged_robot；URDF 在首次构造时转换。
- 入口提醒：Recovery 与 Terrain-Reactive 共用资产。若使用作者 play.py，核对 --eval_pushes 是否保留了原推扰；不是新建两个独立场景。
- 上游：[README](https://github.com/zyicome/Wheel-Legged-Lab/blob/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7/README.md)

## T09 · MickyasTA/wheeled_quadruped_robot

- 固定版本：`bcb4ec233927beb6e8b1806fad4703f52131dcbb`
- 依赖：Isaac Sim 5.1.0 + Isaac Lab 2.3.2.post1 + Python 3.11；README 另列 torch/rsl-rl 依赖
- 安装参考：按 README pip 安装匹配的 Isaac Lab，再 pip install -e source/wheeled_quadruped。
- 入口提醒：必须 Wheeled-Quadruped-Balance-v0，非 Balance-Play-v0；后者移除 push。
- 上游：[README](https://github.com/MickyasTA/wheeled_quadruped_robot/blob/bcb4ec233927beb6e8b1806fad4703f52131dcbb/README.md)

## T10 · BrandoUlissi/isaaclab-go2-locomotion

- 固定版本：`22c0dc900a9a9f38349cc959979e820e41021785`
- 依赖：Isaac Sim 4.5.0 + Isaac Lab 2.1.1；README: rsl_rl 2.3.3
- 安装参考：使用作者 train_pushrecovery wrapper 导入自定义注册，再在 Isaac Lab 环境运行。不要直接只安装官方 Go2 平地任务。
- 入口提醒：随机脉冲版本与作者 scheduled 120 N 演示是两个设置；课程初始最大 30 N，必须记录课程步数/强度。
- 上游：[README](https://github.com/BrandoUlissi/isaaclab-go2-locomotion/blob/22c0dc900a9a9f38349cc959979e820e41021785/README.md)

## T11 · agilexrobotics/robot_lab

- 固定版本：`b868140eeb1459acefef24a865587ec39a5278c3`
- 依赖：Isaac Sim 4.5.0 + Isaac Lab 2.0.0 + Python 3.10
- 安装参考：按 README 安装 source/robot_lab 扩展；作者 scripts/rsl_rl/base 提供训练与可视化入口。
- 入口提醒：RobotLab-Isaac-Velocity-Flat-HandStand-Unitree-A1-v0；默认 back 指后腿抬起、以前腿支撑。
- 上游：[README](https://github.com/agilexrobotics/robot_lab/blob/b868140eeb1459acefef24a865587ec39a5278c3/README.md)

## T12 · TeleHuman/HumanoidSoccer

- 固定版本：`e72e470230047dedaf66df0983f1d0ab746faeb5`
- 依赖：Isaac Lab 2.1.1；对应 Isaac Sim 4.5 系列，按作者 README 安装
- 安装参考：安装 whole_body_tracking 扩展；资产 URDF 在线转换，另准备 motions/soccer-standard。本地只有源码/目录核查，未运行转换。
- 入口提醒：Tracking-Flat-G1-SoccerMoving-RNN-v0；--motion_path motions/soccer-standard；作者 ONNX 是复现参考，不能直接作为被测 agent 的动作工具。
- 上游：[README](https://github.com/TeleHuman/HumanoidSoccer/blob/e72e470230047dedaf66df0983f1d0ab746faeb5/README.md)

## T13 · isaac-sim/IsaacLab

- 固定版本：`b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8`
- 依赖：固定源码 VERSION=2.3.2；README 支持 Sim 4.5/5.0/5.1，建议单独记录实际运行版本
- 安装参考：按该 commit README 安装 Isaac Lab；任务随 isaaclab_tasks 注册，无额外第三方任务包。
- 入口提醒：Isaac-Tracking-LocoManip-Digit-v0；保持原全身动作接口，不调用外部训练好的 Digit walking controller。
- 上游：[README](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/README.md)

## T14 · isaac-sim/OmniIsaacGymEnvs

- 固定版本：`f8f91bcf73cfd8ede2eab70e19dd70162abf9775`
- 依赖：固定 README 要求 Isaac Sim 4.0.0；不是旧独立 Isaac Gym
- 安装参考：使用 Isaac Sim 自带 python.sh；pip install -e .；按 README 的 rlgames_train 入口运行。
- 入口提醒：task=AnymalTerrain；保留原每 15 s push 与地形，20 s episode，记录是否发生推扰。
- 上游：[README](https://github.com/isaac-sim/OmniIsaacGymEnvs/blob/f8f91bcf73cfd8ede2eab70e19dd70162abf9775/README.md)

## T15 · jaykorea/Isaac-RL-Two-wheel-Legged-Bot

- 固定版本：`d922cce9e07a8877c37e12b2cb39dfdcf34b9d40`
- 依赖：Isaac Sim 4.5 + Isaac Lab 2.0.0 + Python 3.10（README）
- 安装参考：按 README 安装 lab.flamingo，手动解压指定资产 ZIP。不要按 README 示例下载另一个 rev 的机器人。
- 入口提醒：Isaac-TrackJUMP-Flat-Flamingo-v1-ppo；非 Play；TrackJump 使用 rev01_5_2，而不是 README 安装示例中的 rev01_4_1。
- 上游：[README](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/d922cce9e07a8877c37e12b2cb39dfdcf34b9d40/README.md)

## T16, T17 · btx0424/OmniDrones

- 固定版本：`9ce7c2028b71be64d7e748c31f685cd3b54afe27`
- 依赖：固定 README 当前分支 Isaac Sim 4.1.0；不是旧 release 分支的 2022.2.0
- 安装参考：按固定 README 和 docs 安装 OmniDrones；两个 task 共享 Hummingbird 模型与同一容器。
- 入口提醒：Hydra task=Payload/PayloadHover 或 task=InvPendulum/InvPendulumTrack；action_transform=null 为本次选择接口。
- 上游：[README](https://github.com/btx0424/OmniDrones/blob/9ce7c2028b71be64d7e748c31f685cd3b54afe27/README.md)
