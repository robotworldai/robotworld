# T01 · 端托盘行走，同时抵抗物体与身体推扰

来源：[AllenHuangGit/SteadyTray](https://github.com/AllenHuangGit/SteadyTray)

固定 commit：`ae781f9f43f9505118c2c434733c57ed1c09dbc3`

原 task/config：`G1-Steady-Object（保留原 robot/object 推扰）`

运行依赖：Isaac Sim 4.5 + 作者 IsaacLab_SteadyTray 分支；不可直接替换成官方同名版本

接入状态：本包未实现或验证 RoboWorld_Scaffold adapter；提供原生任务接入材料。

## 资产与获取

### 带托盘支架的 G1

- 类型：repository
- 位置：`source/steadytray/steadytray/assets/usds/g1_side_tray_holder.usd`
- 核查：本地为 Git LFS 指针；需 git lfs pull
- 定义：[源码](https://github.com/AllenHuangGit/SteadyTray/blob/ae781f9f43f9505118c2c434733c57ed1c09dbc3/source/steadytray/steadytray/assets/robots/g1_delay.py)
- 获取/来源：[上游位置](https://github.com/AllenHuangGit/SteadyTray/blob/ae781f9f43f9505118c2c434733c57ed1c09dbc3/source/steadytray/steadytray/assets/usds/g1_side_tray_holder.usd)
- 配套：保留 assets/usds 整个目录以及 USD 的相对引用
- 注意：这是 env 实际使用的 G1_DELAY_CFG；不要误用同仓库另一份普通 G1 USD。

### 托盘与圆柱负载

- 类型：procedural
- 位置：`steady_tray_env_cfg.py: CuboidCfg；steady_object_env_cfg.py: CylinderCfg`
- 核查：运行时由所列源码生成，没有独立资产包
- 定义：[源码](https://github.com/AllenHuangGit/SteadyTray/blob/ae781f9f43f9505118c2c434733c57ed1c09dbc3/source/steadytray/steadytray/tasks/envs/steady_object_env_cfg.py)
- 获取/来源：[上游位置](https://github.com/AllenHuangGit/SteadyTray/blob/ae781f9f43f9505118c2c434733c57ed1c09dbc3/source/steadytray/steadytray/tasks/envs/steady_object_env_cfg.py)
- 注意：尺寸、质量、摩擦与随机化以固定源码为准。

### 作者学生策略（仅复现参考）

- 类型：repository
- 位置：`model/model_9999.pt`
- 核查：本地为 Git LFS 指针；需 git lfs pull
- 定义：[源码](https://github.com/AllenHuangGit/SteadyTray/blob/ae781f9f43f9505118c2c434733c57ed1c09dbc3/README.md)
- 获取/来源：[上游位置](https://github.com/AllenHuangGit/SteadyTray/blob/ae781f9f43f9505118c2c434733c57ed1c09dbc3/model/model_9999.pt)
- 注意：该模型对应 Stage 4 Distillation，不应直接当作 Stage 3 teacher 权重；不是被测 agent 的免费技能。

## 原生 setting

选作者公开 G1-Steady-Object 环境。物体放在托盘上，未被固定；原配置已有身体推扰与物体推扰，均保持开启。论文还单独报告 Push Robot / Push Object 测试；其每 5 秒施力的协议不能与当前公开环境的速度推扰参数混写。 物体每 2–4 s 施加速度扰动，身体每 3–5 s 推扰；20 s 回合。物理 200 Hz、动作 50 Hz。

## 安装参考

按 README 的作者 fork + Docker 流程安装，再安装 SteadyTray 扩展；依赖 fork 的 termination manager 已附固定版本证据。

## 入口与控制边界

G1-Steady-Object 是原 Stage 3 环境；作者训练 runner 使用残差训练。测试 agent 时，必须记录是接原完整关节动作还是接作者残差 runner，不能混为同一个 setting。

## 动作接口

原环境动作是全身关节位置目标，物理 200 Hz、控制 50 Hz。状态版包含本体与托盘/物体状态；student 版以物体位姿历史减少特权信息。训练论文的 residual 架构是作者的解法，不是复用任务时必须附送的控制能力。

## 观测条件

保留原生观测定义；源码配置中可能包含 simulator state。原生 state 条件与 RGB/RGB-D 条件需分开记录，本包不擅自宣称纯视觉。

## 原始评估与失败

论文 SR 要求物体在整个回合直立并留在托盘上。公开配置记录物体高度低于 0.7 m、倾角超过 0.7 rad；普通配置持续违反 1 秒会触发延迟截断，Play 配置只追踪。应累计这些失败事件，不能只把没有即时 done 当成功。另有速度跟踪误差。

## 接入风险

必须保留原生 robot/object 推扰。物体失败项有 track_only=True 与连续失败时间判定，不能把“episode 没结束”算成功。Play 不等于评测设置。

## 为什么需要新反馈

原身体推扰每 3–5 秒重新采样水平速度（x/y 各 ±0.5 m/s）；物体每 2–4 秒重新采样平移和转动速度（x/y 各 ±0.3 m/s，roll/pitch 各 ±0.3 rad/s）。身体受扰后如何跨步、托盘如何补偿，取决于新发生的倾斜、滑移和物体姿态。

## 反馈为何不能慢

受扰后的身体失衡与载物倾倒会继续发展。物体一旦倒下或滑落，事后恢复机器人姿态也不能满足全程保持直立的任务目标。因此不能等一次慢模型调用结束后才处理受扰状态。

## 难度结论

有公开实机失败/成功对比和原始随机扰动配置，是优先级最高的一题；作者 RL 结果不是 coding agent 成绩。

## 已有视频/轨迹查看记录

实际查看官方实机视频片段：走路端杯，以及踢腿/推物对比；约 7.98 s 处可见 baseline 物体倾倒与所提方法保持。仅抽查片段，不等于复现源码版本。

## 源码定位

- [原生物体推扰、物体反馈与失败记录](https://github.com/AllenHuangGit/SteadyTray/blob/ae781f9f43f9505118c2c434733c57ed1c09dbc3/source/steadytray/steadytray/tasks/envs/steady_object_env_cfg.py#L134) · [包内文件](../../sources/AllenHuangGit/SteadyTray/source/steadytray/steadytray/tasks/envs/steady_object_env_cfg.py)
- [继承的身体推扰与全身动作](https://github.com/AllenHuangGit/SteadyTray/blob/ae781f9f43f9505118c2c434733c57ed1c09dbc3/source/steadytray/steadytray/tasks/envs/locomotion_env_cfg.py#L206) · [包内文件](../../sources/AllenHuangGit/SteadyTray/source/steadytray/steadytray/tasks/envs/locomotion_env_cfg.py)
- [G1-Steady-Object 原任务注册](https://github.com/AllenHuangGit/SteadyTray/blob/ae781f9f43f9505118c2c434733c57ed1c09dbc3/source/steadytray/steadytray/tasks/envs/__init__.py#L37) · [包内文件](../../sources/AllenHuangGit/SteadyTray/source/steadytray/steadytray/tasks/envs/__init__.py)
- [论文：扰动评测与全程不倾倒、不掉落的成功定义](https://arxiv.org/html/2603.10306v1#S4.SS1)
- [作者 Isaac Lab 分支：track_only 与延迟截断的区别](https://github.com/AllenHuangGit/IsaacLab_SteadyTray/blob/3e14846319ac07b5c430c8b6982745a8e7d1d95e/source/isaaclab/isaaclab/managers/termination_manager.py#L196) · [包内文件](../../sources/AllenHuangGit/IsaacLab_SteadyTray/source/isaaclab/isaaclab/managers/termination_manager.py)
- [官方视频与实验](https://steadytray.github.io/)
