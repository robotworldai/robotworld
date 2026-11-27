# T16 · 无人机吊载悬停并抑制受扰摆动

来源：[btx0424/OmniDrones](https://github.com/btx0424/OmniDrones)

固定 commit：`9ce7c2028b71be64d7e748c31f685cd3b54afe27`

原 task/config：`Payload/PayloadHover.yaml`

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

### 吊杆、载荷与关节

- 类型：procedural
- 位置：`omni_drones/envs/payload/utils.py: attach_payload`
- 核查：运行时由所列源码生成，没有独立资产包
- 定义：[源码](https://github.com/btx0424/OmniDrones/blob/9ce7c2028b71be64d7e748c31f685cd3b54afe27/omni_drones/envs/payload/utils.py)
- 获取/来源：[上游位置](https://github.com/btx0424/OmniDrones/blob/9ce7c2028b71be64d7e748c31f685cd3b54afe27/omni_drones/envs/payload/utils.py)

## 原生 setting

原 PayloadHover 存在运行中载荷随机力；push_interval 参数按实现解释，不能直接读成“固定每 2 秒”。

## 安装参考

按固定 README 和 docs 安装 OmniDrones；两个 task 共享 Hummingbird 模型与同一容器。

## 入口与控制边界

Hydra task=Payload/PayloadHover 或 task=InvPendulum/InvPendulumTrack；action_transform=null 为本次选择接口。

## 动作接口

选原 action_transform=null 的转子动作；配置里出现 Lee 控制器名字，不代表已自动调用悬停控制器。

## 观测条件

保留原生观测定义；源码配置中可能包含 simulator state。原生 state 条件与 RGB/RGB-D 条件需分开记录，本包不擅自宣称纯视觉。

## 原始评估与失败

机体或载荷高度 <0.2 m 失败，另有悬停与载荷状态奖励；完整任务达标不能只判断没落地。

## 接入风险

属于空中任务，但与排球、梁球系统有不同受控对象；同系列不同机型不额外计数。

## 为什么需要新反馈

载荷新受力后的摆角/速度与机体状态决定推力分配；仅稳定机体不足以保证载荷稳定。

## 反馈为何不能慢

负载与机体的耦合运动持续发展，失高/大幅摆动需要在碰地前处理。

## 难度结论

运行中新随机载荷力明确，经典耦合控制能否轻松通过尚未验证。

## 已有视频/轨迹查看记录

已核查原始源码；本次未实际观看该固定版本的完整成功/失败视频。

## 源码定位

- [omni_drones/envs/payload/payload_hover.py](https://github.com/btx0424/OmniDrones/blob/9ce7c2028b71be64d7e748c31f685cd3b54afe27/omni_drones/envs/payload/payload_hover.py) · [包内文件](../../sources/btx0424/OmniDrones/omni_drones/envs/payload/payload_hover.py)
