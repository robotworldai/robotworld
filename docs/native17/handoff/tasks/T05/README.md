# T05 · 无人机一对一排球对抗

来源：[thu-uav/VolleyBots](https://github.com/thu-uav/VolleyBots)

固定 commit：`10e3701e480b518041c8be6a40efee7bdcb69283`

原 task/config：`Volleyball1v1`

运行依赖：Isaac Sim 2023.1.0-hotfix.1 + 作者 Orbit/依赖；独立旧版本容器

接入状态：本包未实现或验证 RoboWorld_Scaffold adapter；提供原生任务接入材料。

## 资产与获取

### Iris 无人机

- 类型：repository
- 位置：`volley_bots/robots/assets/usd/iris.usd`
- 核查：已核对本地仓库文件；未在 Isaac 中加载
- 定义：[源码](https://github.com/thu-uav/VolleyBots/blob/10e3701e480b518041c8be6a40efee7bdcb69283/volley_bots/robots/drone/iris.py)
- 获取/来源：[上游位置](https://github.com/thu-uav/VolleyBots/blob/10e3701e480b518041c8be6a40efee7bdcb69283/volley_bots/robots/assets/usd/iris.usd)
- 配套：同目录 iris.yaml 与 USD 引用的 mesh/material
- 注意：Volleyball1v1.yaml 的 drone_model=Iris；不是 IrisRacket 名称的另一个模型。

### 本地地面背景（启用 use_local_usd 时）

- 类型：repository
- 位置：`volley_bots/envs/assets/default_environment.usd`
- 核查：已核对本地仓库文件；未在 Isaac 中加载
- 定义：[源码](https://github.com/thu-uav/VolleyBots/blob/10e3701e480b518041c8be6a40efee7bdcb69283/volley_bots/envs/competitive/volleyball_1v1.py)
- 获取/来源：[上游位置](https://github.com/thu-uav/VolleyBots/blob/10e3701e480b518041c8be6a40efee7bdcb69283/volley_bots/envs/assets/default_environment.usd)
- 注意：关闭 use_local_usd 时走 Isaac 的在线 ground-plane 资源。

### 排球与球网/场地几何

- 类型：procedural
- 位置：`volleyball_1v1.py 的 _design_scene 及 helpers`
- 核查：运行时由所列源码生成，没有独立资产包
- 定义：[源码](https://github.com/thu-uav/VolleyBots/blob/10e3701e480b518041c8be6a40efee7bdcb69283/volley_bots/envs/competitive/volleyball_1v1.py)
- 获取/来源：[上游位置](https://github.com/thu-uav/VolleyBots/blob/10e3701e480b518041c8be6a40efee7bdcb69283/volley_bots/envs/competitive/volleyball_1v1.py)
- 注意：对手策略 checkpoint 另从仓库 checkpoints/ 挑选并固定；未替智钦预选对手强度。

## 原生 setting

一个 agent 控制本方无人机，对手独立运行自己的策略；保留原物理与比赛规则。不能让同一个待测 agent 预先指定双方整场配合轨迹。

## 安装参考

README 提供 jimmyzhangruize/isaac-sim:2023.1.0-hotfix.1 镜像；需按 README mount 克隆目录。镜像未在本机拉取。

## 入口与控制边界

Volleyball1v1；锁定对手与 checkpoint。checkpoint 目录有作者基线，先核对所用文件，不把分层 Serve/Attack 技能接为 agent 免费小脑。

## 动作接口

原生默认是旋翼推力；框架还支持 CTBR（总推力+机体角速度）转换。使用 CTBR 时，底层飞控不能额外提供自动接球策略。

## 观测条件

保留原生观测定义；源码配置中可能包含 simulator state。原生 state 条件与 RGB/RGB-D 条件需分开记录，本包不擅自宣称纯视觉。

## 原始评估与失败

原代码包含落地、出界、触网、错误击球等判定并记录胜负；可直接定义固定对手集的胜率，但必须公开对手版本。

## 接入风险

固定对手 checkpoint、场地、动作层级与种子。直接调用作者 Serve/Attack 技能会把小脑交给现成策略。1v1、3v3、6v6 不作为三个不同机制凑数。

## 为什么需要新反馈

对方击球会改变来球方向和速度；下一个接球位置取决于对手刚刚产生的动作与碰撞结果。需要在每次回球后读取新的球态与对手状态。

## 反馈为何不能慢

来球落到本方场内就丢分，击球窗口过去不能补做。规则还限制单次回球与越网/出界，不允许把球接住后慢慢商量。

## 难度结论

对方击球后产生新的球轨迹，存在落地/出界截止；比固定节奏单机颠球的反馈必要性更清楚。

## 已有视频/轨迹查看记录

实际打开并播放官网 1v1.mp4（20 s），查看了开场画面；未完整逐回合判读，未取得原始状态日志。论文胜率与分层策略结果属于 RL/多智能体设置，不是本项目结果。

## 源码定位

- [一对一原生任务配置](https://github.com/thu-uav/VolleyBots/blob/10e3701e480b518041c8be6a40efee7bdcb69283/cfg/task/Volleyball1v1.yaml#L1) · [包内文件](../../sources/thu-uav/VolleyBots/cfg/task/Volleyball1v1.yaml)
- [真实接触、轮次与比赛结算](https://github.com/thu-uav/VolleyBots/blob/10e3701e480b518041c8be6a40efee7bdcb69283/volley_bots/envs/competitive/volleyball_1v1.py#L944) · [包内文件](../../sources/thu-uav/VolleyBots/volley_bots/envs/competitive/volleyball_1v1.py)
- [落地、出界、犯规判据](https://github.com/thu-uav/VolleyBots/blob/10e3701e480b518041c8be6a40efee7bdcb69283/volley_bots/envs/competitive/utils/rules.py#L93) · [包内文件](../../sources/thu-uav/VolleyBots/volley_bots/envs/competitive/utils/rules.py)
- [官方任务和视频](https://volleybots.github.io/)
