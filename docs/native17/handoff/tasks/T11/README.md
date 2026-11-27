# T11 · A1 抬起后腿、以前腿支撑并抗扰

来源：[agilexrobotics/robot_lab](https://github.com/agilexrobotics/robot_lab)

固定 commit：`b868140eeb1459acefef24a865587ec39a5278c3`

原 task/config：`RobotLab-Isaac-Velocity-Flat-HandStand-Unitree-A1-v0`

运行依赖：Isaac Sim 4.5.0 + Isaac Lab 2.0.0 + Python 3.10

接入状态：本包未实现或验证 RoboWorld_Scaffold adapter；提供原生任务接入材料。

## 资产与获取

### Unitree A1

- 类型：repository
- 位置：`source/robot_lab/data/Robots/Unitree/A1/a1.usd`
- 核查：配置/上游目录指向该位置；本轮未下载资产本体
- 定义：[源码](https://github.com/agilexrobotics/robot_lab/blob/b868140eeb1459acefef24a865587ec39a5278c3/source/robot_lab/robot_lab/assets/unitree.py)
- 获取/来源：[上游位置](https://github.com/agilexrobotics/robot_lab/blob/b868140eeb1459acefef24a865587ec39a5278c3/source/robot_lab/data/Robots/Unitree/A1/a1.usd)
- 注意：本项目 ISAACLAB_ASSETS_DATA_DIR 指向 source/robot_lab/data，不是线上 Nucleus 根目录。

## 原生 setting

默认 handstand_type="back" 是后腿抬起、前腿支撑。抬脚目标 0.5 m；20 s 回合，继承 10–15 s 随机平面推速 ±0.5。

## 安装参考

按 README 安装 source/robot_lab 扩展；作者 scripts/rsl_rl/base 提供训练与可视化入口。

## 入口与控制边界

RobotLab-Isaac-Velocity-Flat-HandStand-Unitree-A1-v0；默认 back 指后腿抬起、以前腿支撑。

## 动作接口

全身关节位置残差加 PD；没有内置手倒立策略。

## 观测条件

保留原生观测定义；源码配置中可能包含 simulator state。原生 state 条件与 RGB/RGB-D 条件需分开记录，本包不擅自宣称纯视觉。

## 原始评估与失败

原奖励分别评价抬脚高度、姿态与其他行走项，没有现成“全程手倒立成功率”；不能只用未触发 done 充当完成。

## 接入风险

确实继承 push，不把 reset 的力随机化误写成运行中持续力。新 SR 需组合原目标项与跌倒记录并单独披露。

## 为什么需要新反馈

窄支撑下躯干倾角/角速度决定如何调整支撑脚与身体；推扰后需要新姿态反馈。

## 反馈为何不能慢

身体绕有限支撑区域倾倒会失去恢复机会，不能依靠长时间保持旧关节目标；原动作周期 20 ms。

## 难度结论

前腿双点支撑比普通静态四足站立更能暴露全身控制；缺少本项目基线分数。

## 已有视频/轨迹查看记录

源码核查；未观看精确 Flat-HandStand 的成功/失败演示。

## 源码定位

- [倒立姿态和配置继承](https://github.com/agilexrobotics/robot_lab/blob/b868140eeb1459acefef24a865587ec39a5278c3/source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/unitree_a1_handstand/rough_env_cfg.py) · [包内文件](../../sources/agilexrobotics/robot_lab/source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/unitree_a1_handstand/rough_env_cfg.py)
- [平地版本](https://github.com/agilexrobotics/robot_lab/blob/b868140eeb1459acefef24a865587ec39a5278c3/source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/unitree_a1_handstand/flat_env_cfg.py) · [包内文件](../../sources/agilexrobotics/robot_lab/source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/unitree_a1_handstand/flat_env_cfg.py)
- [继承推扰](https://github.com/agilexrobotics/robot_lab/blob/b868140eeb1459acefef24a865587ec39a5278c3/source/robot_lab/robot_lab/tasks/locomotion/velocity/velocity_env_cfg.py#L362) · [包内文件](../../sources/agilexrobotics/robot_lab/source/robot_lab/robot_lab/tasks/locomotion/velocity/velocity_env_cfg.py)
