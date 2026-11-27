# T17 · 无人机托举倒立摆并跟踪轨迹

来源：[btx0424/OmniDrones](https://github.com/btx0424/OmniDrones)

固定 commit：`9ce7c2028b71be64d7e748c31f685cd3b54afe27`

原 task/config：`InvPendulum/InvPendulumTrack.yaml`

运行依赖：固定 README 当前分支 Isaac Sim 4.1.0；不是旧 release 分支的 2022.2.0

接入状态：本包未实现或验证 RoboWorld_Scaffold adapter；提供原生任务接入材料。

## 资产与获取

### Hummingbird 四旋翼

- 类型：repository
- 位置：`omni_drones/robots/assets/usd/hummingbird.usd`
- 核查：配置/上游目录指向该位置；本轮未下载资产本体
- 定义：[源码](https://github.com/btx0424/OmniDrones/blob/9ce7c2028b71be64d7e748c31f685cd3b54afe27/omni_drones/robots/drone/hummingbird.py)
- 获取/来源：[上游位置](https://github.com/btx0424/OmniDrones/blob/9ce7c2028b71be64d7e748c31f685cd3b54afe27/omni_drones/robots/assets/usd/hummingbird.usd)
- 配套：同目录 hummingbird.yaml 以及 referenced assets
- 注意：两个原 task 都配置 Hummingbird，不是 Iris；local cache 本次主要为代码，需从原仓库拿完整 USD。

### 上置摆杆、摆球与关节

- 类型：procedural
- 位置：`omni_drones/envs/inv_pendulum/utils.py: create_pendulum`
- 核查：运行时由所列源码生成，没有独立资产包
- 定义：[源码](https://github.com/btx0424/OmniDrones/blob/9ce7c2028b71be64d7e748c31f685cd3b54afe27/omni_drones/envs/inv_pendulum/utils.py)
- 获取/来源：[上游位置](https://github.com/btx0424/OmniDrones/blob/9ce7c2028b71be64d7e748c31f685cd3b54afe27/omni_drones/envs/inv_pendulum/utils.py)

## 原生 setting

原 InvPendulumTrack：1 m 上置杆、600 step 回合；参考轨迹已知并有未来参考采样。没有额外添加风扰。

## 安装参考

按固定 README 和 docs 安装 OmniDrones；两个 task 共享 Hummingbird 模型与同一容器。

## 入口与控制边界

Hydra task=Payload/PayloadHover 或 task=InvPendulum/InvPendulumTrack；action_transform=null 为本次选择接口。

## 动作接口

原生飞行动作入口，不能先接上完整倒立摆稳定策略再仅评高层指令。

## 观测条件

保留原生观测定义；源码配置中可能包含 simulator state。原生 state 条件与 RGB/RGB-D 条件需分开记录，本包不擅自宣称纯视觉。

## 原始评估与失败

沿用原终止与跟踪误差；同时检查杆姿态，不能只评机体到达轨迹点。

## 接入风险

没有运行中新随机力的强证据，分类依赖不稳定动力学；确定初始状态下的短时开环轨迹究竟能撑多久，必须跑对照。

## 为什么需要新反馈

上置摆的实际偏角和角速度必须决定机体纠偏，已知期望轨迹不等于已知实际摆状态。

## 反馈为何不能慢

摆杆偏离会继续倾倒；杆向上分量 <0.2、机体过低或轨迹误差 >0.8 即失败。

## 难度结论

保留控制机制；证据强度低于带明确随机推扰的任务，不承诺每个 seed 都无法开环通过。

## 已有视频/轨迹查看记录

已核查原始源码；本次未实际观看该固定版本的完整成功/失败视频。

## 源码定位

- [omni_drones/envs/inv_pendulum/inv_pendulum_track.py](https://github.com/btx0424/OmniDrones/blob/9ce7c2028b71be64d7e748c31f685cd3b54afe27/omni_drones/envs/inv_pendulum/inv_pendulum_track.py) · [包内文件](../../sources/btx0424/OmniDrones/omni_drones/envs/inv_pendulum/inv_pendulum_track.py)
