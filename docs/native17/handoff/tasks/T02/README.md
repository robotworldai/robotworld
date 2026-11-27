# T02 · 人形机器人连续接回随机乒乓来球

来源：[purdue-tracelab/TTRL-ICRA2026](https://github.com/purdue-tracelab/TTRL-ICRA2026)

固定 commit：`fdb192f8fff9f5ca54c4ec4a8441fda5174ca078`

原 task/config：`t1_tt_eval`

运行依赖：Isaac Sim 4.5.0 + Isaac Lab 2.1.0；作者提示升级 Sim 5.0+ 会改变成绩

接入状态：本包未实现或验证 RoboWorld_Scaffold adapter；提供原生任务接入材料。

## 资产与获取

### 带乒乓配置的 Booster T1

- 类型：repository
- 位置：`legged_lab/assets/booster/T1_TT/T1_TT.usd`
- 核查：已核对本地仓库文件；未在 Isaac 中加载
- 定义：[源码](https://github.com/purdue-tracelab/TTRL-ICRA2026/blob/fdb192f8fff9f5ca54c4ec4a8441fda5174ca078/legged_lab/assets/booster/booster.py)
- 获取/来源：[上游位置](https://github.com/purdue-tracelab/TTRL-ICRA2026/blob/fdb192f8fff9f5ca54c4ec4a8441fda5174ca078/legged_lab/assets/booster/T1_TT/T1_TT.usd)
- 注意：t1_tt_eval 使用 BOOSTER_T1_TT_P2_CFG；保留同目录 mesh/material 引用。

### 乒乓球桌

- 类型：repository
- 位置：`legged_lab/assets/table_tennis/table/pp_table_ver2.usd`
- 核查：已核对本地仓库文件；未在 Isaac 中加载
- 定义：[源码](https://github.com/purdue-tracelab/TTRL-ICRA2026/blob/fdb192f8fff9f5ca54c4ec4a8441fda5174ca078/legged_lab/assets/table_tennis/table.py)
- 获取/来源：[上游位置](https://github.com/purdue-tracelab/TTRL-ICRA2026/blob/fdb192f8fff9f5ca54c4ec4a8441fda5174ca078/legged_lab/assets/table_tennis/table/pp_table_ver2.usd)

### 球及预测位置标记

- 类型：procedural
- 位置：`SphereCfg / legged_lab/assets/table_tennis/ball.py`
- 核查：运行时由所列源码生成，没有独立资产包
- 定义：[源码](https://github.com/purdue-tracelab/TTRL-ICRA2026/blob/fdb192f8fff9f5ca54c4ec4a8441fda5174ca078/legged_lab/envs/base/tt_env.py)
- 获取/来源：[上游位置](https://github.com/purdue-tracelab/TTRL-ICRA2026/blob/fdb192f8fff9f5ca54c4ec4a8441fda5174ca078/legged_lab/envs/base/tt_env.py)
- 注意：预测器权重随作者训练 checkpoint 保存；本轮未拿到匹配 checkpoint。

## 原生 setting

保留原多次发球逻辑与随机来球速度；使用完整的人形机器人关节控制，不能把作者整套乒乓球策略包装成 agent 的工具。

## 安装参考

按 README 安装 legged_lab 与作者 runner；先核对资产，再用 t1_tt_eval。

## 入口与控制边界

t1_tt_eval 为作者任务名；直接从注册表构造。--predictor 是作者额外预测器，不应悄悄赠送给 agent；若使用需标注观测条件。

## 动作接口

直接输出动作关节的位置目标；物理步长 0.002 s，decimation=10，即原策略更新 50 Hz。这个频率是源码配置，不是已实测的最低必要频率。

## 观测条件

原机器人/球状态及作者 predictor 的使用方式要分别记录；本包不将 cuRobo/球预测器/全身策略自动作为 agent 可用工具。

## 原始评估与失败

reward_table_success 使用“已经碰过球拍”和“刚接触对面球台”；身体过低或越界终止。复用回球成功事件统计每次发球的回球率；不能把 dense reward 直接叫任务成功率。

## 接入风险

保留 t1_tt_eval 的来球分布与重发机制。不能把“碰到球”当成回球成功，也不能向 agent 提供现成乒乓 policy。

## 为什么需要新反馈

同一身体回合内，球落地或球回合超时后只重置球，新球速度重新采样。之后的来球不包含在第一张观测里；需要根据新来球决定移动、挥拍与击球时机。

## 反馈为何不能慢

评测配置的来球 x 方向速度为 −6.5 至 −5.2 m/s；球越过可击区域就错失这一球。挥拍时还要保持身体稳定。等待模型慢慢处理，不能把已经错过的来球补回来。

## 难度结论

来球在一个身体 episode 中重新采样，反馈截止时间明确；全身平衡与回球耦合。具体难度需要同接口基线验证。

## 已有视频/轨迹查看记录

找到作者仿真片段和 YouTube 演示入口；本次没有完成视频观看，不将其记作已观测的成功轨迹。

## 源码定位

- [连续回合中重新采样并发球](https://github.com/purdue-tracelab/TTRL-ICRA2026/blob/fdb192f8fff9f5ca54c4ec4a8441fda5174ca078/legged_lab/envs/base/tt_env.py#L657) · [包内文件](../../sources/purdue-tracelab/TTRL-ICRA2026/legged_lab/envs/base/tt_env.py)
- [原评测来球速度](https://github.com/purdue-tracelab/TTRL-ICRA2026/blob/fdb192f8fff9f5ca54c4ec4a8441fda5174ca078/legged_lab/envs/t1_tt/t1_tt_config.py#L296) · [包内文件](../../sources/purdue-tracelab/TTRL-ICRA2026/legged_lab/envs/t1_tt/t1_tt_config.py)
- [实际回球到对面球台的判定](https://github.com/purdue-tracelab/TTRL-ICRA2026/blob/fdb192f8fff9f5ca54c4ec4a8441fda5174ca078/legged_lab/mdp/rewards.py#L613) · [包内文件](../../sources/purdue-tracelab/TTRL-ICRA2026/legged_lab/mdp/rewards.py)
- [作者演示入口](https://github.com/purdue-tracelab/TTRL-ICRA2026)
