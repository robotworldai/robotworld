# T14 · 四足崎岖地形行走并抵抗推扰

来源：[isaac-sim/OmniIsaacGymEnvs](https://github.com/isaac-sim/OmniIsaacGymEnvs)

固定 commit：`f8f91bcf73cfd8ede2eab70e19dd70162abf9775`

原 task/config：`AnymalTerrain`

运行依赖：固定 README 要求 Isaac Sim 4.0.0；不是旧独立 Isaac Gym

接入状态：本包未实现或验证 RoboWorld_Scaffold adapter；提供原生任务接入材料。

## 资产与获取

### ANYmal

- 类型：external
- 位置：`get_assets_root_path() + /Isaac/Robots/ANYbotics/anymal_instanceable.usd`
- 核查：外部资产；需由目标 Isaac 安装解析/下载，未验证远端下载
- 定义：[源码](https://github.com/isaac-sim/OmniIsaacGymEnvs/blob/f8f91bcf73cfd8ede2eab70e19dd70162abf9775/omniisaacgymenvs/robots/articulations/anymal.py)
- 注意：使用 Isaac Sim 4.0 的资产服务或缓存根，保留 instanceable 引用。

### 崎岖地形

- 类型：procedural
- 位置：`AnymalTerrain 配置与 terrain_utils`
- 核查：运行时由所列源码生成，没有独立资产包
- 定义：[源码](https://github.com/isaac-sim/OmniIsaacGymEnvs/blob/f8f91bcf73cfd8ede2eab70e19dd70162abf9775/omniisaacgymenvs/tasks/anymal_terrain.py)
- 获取/来源：[上游位置](https://github.com/isaac-sim/OmniIsaacGymEnvs/blob/f8f91bcf73cfd8ede2eab70e19dd70162abf9775/omniisaacgymenvs/tasks/anymal_terrain.py)

## 原生 setting

AnymalTerrain 原配置为 20 s episode；运行中每 15 s 改变水平基座速度，地形影响落脚与身体状态。

## 安装参考

使用 Isaac Sim 自带 python.sh；pip install -e .；按 README 的 rlgames_train 入口运行。

## 入口与控制边界

task=AnymalTerrain；保留原每 15 s push 与地形，20 s episode，记录是否发生推扰。

## 动作接口

关节位置目标与 PD；不是“前进一米”这种已含行走控制的 API。

## 观测条件

保留原生观测定义；源码配置中可能包含 simulator state。原生 state 条件与 RGB/RGB-D 条件需分开记录，本包不擅自宣称纯视觉。

## 原始评估与失败

身体/膝部接触等终止项，另有速度跟踪奖励；站住不倒并不完成行走任务。

## 接入风险

旧 OmniIsaacGymEnvs 基于 Isaac Sim，区别于独立旧 IsaacGymEnvs。需按旧依赖容器复现，不能直接假定兼容当前 Isaac Lab。

## 为什么需要新反馈

脚与地形接触及推扰后的速度必须进入下一步关节控制；预设步态不能对运行中新扰动反应。

## 反馈为何不能慢

走动中的支撑变化与失衡存在时间窗口；如果 agent 获得作者训练好的 gait，快反馈已经被外包。

## 难度结论

机制清楚但属于成熟 locomotion 问题；和轮足地形题需控制权重，避免排行榜被步态题主导。

## 已有视频/轨迹查看记录

已核查原始源码；本次未实际观看该固定版本的完整成功/失败视频。

## 源码定位

- [omniisaacgymenvs/tasks/anymal_terrain.py](https://github.com/isaac-sim/OmniIsaacGymEnvs/blob/f8f91bcf73cfd8ede2eab70e19dd70162abf9775/omniisaacgymenvs/tasks/anymal_terrain.py) · [包内文件](../../sources/isaac-sim/OmniIsaacGymEnvs/omniisaacgymenvs/tasks/anymal_terrain.py)
