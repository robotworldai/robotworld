# T13 · Digit 行走时同时跟踪双手目标

来源：[isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)

固定 commit：`b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8`

原 task/config：`Isaac-Tracking-LocoManip-Digit-v0`

运行依赖：固定源码 VERSION=2.3.2；README 支持 Sim 4.5/5.0/5.1，建议单独记录实际运行版本

接入状态：本包未实现或验证 RoboWorld_Scaffold adapter；提供原生任务接入材料。

## 资产与获取

### Digit v4

- 类型：external
- 位置：`{NUCLEUS_ASSET_ROOT_DIR}/Isaac/Robots/Agility/Digit/digit_v4.usd`
- 核查：外部资产；需由目标 Isaac 安装解析/下载，未验证远端下载
- 定义：[源码](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab_assets/isaaclab_assets/robots/agility.py)
- 注意：NUCLEUS_ASSET_ROOT_DIR 由 /persistent/isaac/asset_root/cloud 设置决定；补充的 assets.py 给出解析位置。

### 平地与双手目标

- 类型：procedural
- 位置：`DigitRoughEnvCfg → loco_manip_env_cfg.py 的目标/场景覆盖`
- 核查：运行时由所列源码生成，没有独立资产包
- 定义：[源码](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/tracking/config/digit/loco_manip_env_cfg.py)
- 获取/来源：[上游位置](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/tracking/config/digit/loco_manip_env_cfg.py)
- 注意：无被手握住的货物；这是双手目标跟踪，不是搬箱。

## 原生 setting

14 s 回合，双手目标每 1–3 s 更新；继承身体推扰间隔 10–15 s，因此不是每个回合都保证被推。

## 安装参考

按该 commit README 安装 Isaac Lab；任务随 isaaclab_tasks 注册，无额外第三方任务包。

## 入口与控制边界

Isaac-Tracking-LocoManip-Digit-v0；保持原全身动作接口，不调用外部训练好的 Digit walking controller。

## 动作接口

全身关节目标；底层电机伺服不等于已经解决全身平衡。

## 观测条件

保留原生观测定义；源码配置中可能包含 simulator state。原生 state 条件与 RGB/RGB-D 条件需分开记录，本包不擅自宣称纯视觉。

## 原始评估与失败

原身体接触/姿态失败与行走、手目标跟踪指标；没有等价于“搬箱成功”的终局判据。

## 接入风险

原题没有被手握住的货物，不能写成端箱行走；代码里未绑定的手部随机施力不算生效事件。

## 为什么需要新反馈

双手到达误差、脚接触与躯干状态共同影响下一步全身动作。手臂移动本身扰动平衡。

## 反馈为何不能慢

双足动态支撑要在失衡前纠正；不能让机器人维持不稳定支撑等待秒级模型回答。

## 难度结论

保留全身协调机制，先验证手目标与行走都达标；不应仅凭名字 LocoManip 宣称复杂物体接触。

## 已有视频/轨迹查看记录

已核查原始源码；本次未实际观看该固定版本的完整成功/失败视频。

## 源码定位

- [source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/tracking/config/digit/loco_manip_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/tracking/config/digit/loco_manip_env_cfg.py) · [包内文件](../../sources/isaac-sim/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/tracking/config/digit/loco_manip_env_cfg.py)
- [source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/digit/rough_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/digit/rough_env_cfg.py) · [包内文件](../../sources/isaac-sim/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/digit/rough_env_cfg.py)
- [source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py) · [包内文件](../../sources/isaac-sim/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)
