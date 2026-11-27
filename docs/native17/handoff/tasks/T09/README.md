# T09 · 四足机器人以后两轮直立并抵抗随机推扰

来源：[MickyasTA/wheeled_quadruped_robot](https://github.com/MickyasTA/wheeled_quadruped_robot)

固定 commit：`bcb4ec233927beb6e8b1806fad4703f52131dcbb`

原 task/config：`Wheeled-Quadruped-Balance-v0（非 Play）`

运行依赖：Isaac Sim 5.1.0 + Isaac Lab 2.3.2.post1 + Python 3.11；README 另列 torch/rsl-rl 依赖

接入状态：本包未实现或验证 RoboWorld_Scaffold adapter；提供原生任务接入材料。

## 资产与获取

### 以后轮支撑的四足轮式机器人

- 类型：repository
- 位置：`source/wheeled_quadruped/wheeled_quadruped/assets/quadruped_robot.usd`
- 核查：已核对本地仓库文件；未在 Isaac 中加载
- 定义：[源码](https://github.com/MickyasTA/wheeled_quadruped_robot/blob/bcb4ec233927beb6e8b1806fad4703f52131dcbb/source/wheeled_quadruped/wheeled_quadruped/assets/__init__.py)
- 获取/来源：[上游位置](https://github.com/MickyasTA/wheeled_quadruped_robot/blob/bcb4ec233927beb6e8b1806fad4703f52131dcbb/source/wheeled_quadruped/wheeled_quadruped/assets/quadruped_robot.usd)
- 注意：前轮固定、后轮+前大腿共四个动作；不要当作四条自由腿。

### 作者 balance checkpoint（仅复现参考）

- 类型：repository
- 位置：`pretrained/balance`
- 核查：已核对本地目录；未递归验证每个 USD 引用
- 定义：[源码](https://github.com/MickyasTA/wheeled_quadruped_robot/blob/bcb4ec233927beb6e8b1806fad4703f52131dcbb/README.md)
- 获取/来源：[上游位置](https://github.com/MickyasTA/wheeled_quadruped_robot/blob/bcb4ec233927beb6e8b1806fad4703f52131dcbb/pretrained/balance)
- 注意：保留基线来源，不自动接为 agent 工具。

## 原生 setting

选原普通 Balance 配置，不选会关闭推扰的 Balance-Play。任务是机器人像平衡车一样在两只后轮上直立，不是四轮着地行驶。

## 安装参考

按 README pip 安装匹配的 Isaac Lab，再 pip install -e source/wheeled_quadruped。

## 入口与控制边界

必须 Wheeled-Quadruped-Balance-v0，非 Balance-Play-v0；后者移除 push。

## 动作接口

2 个前腿关节位置目标 + 2 个后轮速度目标。原物理 200 Hz、动作更新 50 Hz；actor 输入包括角速度、重力方向、关节状态与上次动作。

## 观测条件

保留原生观测定义；源码配置中可能包含 simulator state。原生 state 条件与 RGB/RGB-D 条件需分开记录，本包不擅自宣称纯视觉。

## 原始评估与失败

原实现是存活/姿态/高度等奖励与摔倒终止，回合 20 秒；没有发布 RobotWorld 所需的统一二元成功率。可复用终止事件，但二元成功口径应明确为我们的评测定义。

## 接入风险

原 Balance 的推扰必须保留；Balance-Play 将 push 关闭。前腿姿态与后轮配合，不是已有底盘平衡器上的 move_to。

## 为什么需要新反馈

回合运行 10–15 秒时会触发新的水平速度推扰，x/y 各在 −0.5 至 0.5 m/s 范围采样。推扰后的倾角和角速度决定需要朝哪边、多快地移动轮子。

## 反馈为何不能慢

两轮直立缺少四足支撑，倾倒会继续发展；超过 60° 或身体高度低于 0.4 m 就结束。保持上一条轮速命令、等待模型再回答，并不能保持受扰后的平衡。

## 难度结论

机构本身不稳定且有运行中推扰；可能被较短经典控制器解决，仍需测难度。

## 已有视频/轨迹查看记录

找到官方 pretrained 与 balance demo 入口；本次未实际观看完整视频。

## 源码定位

- [运行中推扰及后续终止/动作设置](https://github.com/MickyasTA/wheeled_quadruped_robot/blob/bcb4ec233927beb6e8b1806fad4703f52131dcbb/source/wheeled_quadruped/wheeled_quadruped/tasks/balance/balance_env_cfg.py#L185) · [包内文件](../../sources/MickyasTA/wheeled_quadruped_robot/source/wheeled_quadruped/wheeled_quadruped/tasks/balance/balance_env_cfg.py)
- [普通与 Play 任务注册](https://github.com/MickyasTA/wheeled_quadruped_robot/blob/bcb4ec233927beb6e8b1806fad4703f52131dcbb/source/wheeled_quadruped/wheeled_quadruped/tasks/balance/__init__.py#L18) · [包内文件](../../sources/MickyasTA/wheeled_quadruped_robot/source/wheeled_quadruped/wheeled_quadruped/tasks/balance/__init__.py)
