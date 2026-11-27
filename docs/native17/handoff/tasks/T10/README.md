# T10 · 四足机器人行走时抵抗随机外力脉冲

来源：[BrandoUlissi/isaaclab-go2-locomotion](https://github.com/BrandoUlissi/isaaclab-go2-locomotion)

固定 commit：`22c0dc900a9a9f38349cc959979e820e41021785`

原 task/config：`Isaac-Velocity-Flat-Unitree-Go2-PushRecovery-v0`

运行依赖：Isaac Sim 4.5.0 + Isaac Lab 2.1.1；README: rsl_rl 2.3.3

接入状态：本包未实现或验证 RoboWorld_Scaffold adapter；提供原生任务接入材料。

## 资产与获取

### Unitree Go2

- 类型：external
- 位置：`Isaac Lab v2.1.1 的 UNITREE_GO2_CFG → {ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd`
- 核查：外部资产；需由目标 Isaac 安装解析/下载，未验证远端下载
- 定义：[源码](https://github.com/BrandoUlissi/isaaclab-go2-locomotion/blob/22c0dc900a9a9f38349cc959979e820e41021785/src/isaaclab_go2_pushrecovery/env_cfg.py)
- 注意：Go2 继承官方平地机器人配置；资产路径定义另附 v2.1.1 的 unitree.py。

## 原生 setting

原随机脉冲间隔 6–10 s、持续 0.15–0.25 s；力方向随机。课程从最大 30 N 增到 120 N，刚创建环境时不是 120 N。

## 安装参考

使用作者 train_pushrecovery wrapper 导入自定义注册，再在 Isaac Lab 环境运行。不要直接只安装官方 Go2 平地任务。

## 入口与控制边界

随机脉冲版本与作者 scheduled 120 N 演示是两个设置；课程初始最大 30 N，必须记录课程步数/强度。

## 动作接口

关节位置目标加电机级 PD；agent 的程序负责全身控制，不调用已有 Go2 行走策略。

## 观测条件

保留原生观测定义；源码配置中可能包含 simulator state。原生 state 条件与 RGB/RGB-D 条件需分开记录，本包不擅自宣称纯视觉。

## 原始评估与失败

原有跌倒终止与速度跟踪。作者另有固定 120 N 的 scheduled-push 评测脚本，不能把该脚本成绩写成随机训练设置成绩。

## 接入风险

Play 继承普通平地 Go2 配置，可能没有这些外力；20 s 回合也未必触发 25–40 s 间隔的持续力。只依赖已确认会生效的脉冲。

## 为什么需要新反馈

外力造成实际身体速度与支撑脚变化，需要以反馈选择补偿步态，而非重复同一个预排步态。

## 反馈为何不能慢

冲量作用后身体仍在运动，必须在失去可恢复支撑前纠正；脉冲宽度说明事件短，不等于测得的模型反馈截止时间。

## 难度结论

推扰强度会随训练课程变化，难度不能只看 120 N 宣传数字。保留任务，强度/课程状态必须锁定。

## 已有视频/轨迹查看记录

读取作者 release 与评测脚本，未播放 release 中的 mp4；作者报告的对比不等于本次亲自观察或复测。

## 源码定位

- [配置与课程](https://github.com/BrandoUlissi/isaaclab-go2-locomotion/blob/22c0dc900a9a9f38349cc959979e820e41021785/src/isaaclab_go2_pushrecovery/env_cfg.py) · [包内文件](../../sources/BrandoUlissi/isaaclab-go2-locomotion/src/isaaclab_go2_pushrecovery/env_cfg.py)
- [施力实现](https://github.com/BrandoUlissi/isaaclab-go2-locomotion/blob/22c0dc900a9a9f38349cc959979e820e41021785/src/isaaclab_go2_pushrecovery/push_events.py) · [包内文件](../../sources/BrandoUlissi/isaaclab-go2-locomotion/src/isaaclab_go2_pushrecovery/push_events.py)
- [固定推扰评测与原随机环境的差异](https://github.com/BrandoUlissi/isaaclab-go2-locomotion/blob/22c0dc900a9a9f38349cc959979e820e41021785/scripts/play_scheduled_pushes.py) · [包内文件](../../sources/BrandoUlissi/isaaclab-go2-locomotion/scripts/play_scheduled_pushes.py)
- [作者发布记录](https://github.com/BrandoUlissi/isaaclab-go2-locomotion/releases/tag/v0.2.0-pushrecovery)
