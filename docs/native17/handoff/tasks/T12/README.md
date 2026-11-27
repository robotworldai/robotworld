# T12 · 人形机器人抗扰平衡并踢移动足球

来源：[TeleHuman/HumanoidSoccer](https://github.com/TeleHuman/HumanoidSoccer)

固定 commit：`e72e470230047dedaf66df0983f1d0ab746faeb5`

原 task/config：`Tracking-Flat-G1-SoccerMoving-RNN-v0`

运行依赖：Isaac Lab 2.1.1；对应 Isaac Sim 4.5 系列，按作者 README 安装

接入状态：本包未实现或验证 RoboWorld_Scaffold adapter；提供原生任务接入材料。

## 资产与获取

### G1 URDF

- 类型：repository
- 位置：`source/whole_body_tracking/soccer/assets/unitree_description/urdf/g1/main.urdf`
- 核查：配置/上游目录指向该位置；本轮未下载资产本体
- 定义：[源码](https://github.com/TeleHuman/HumanoidSoccer/blob/e72e470230047dedaf66df0983f1d0ab746faeb5/source/whole_body_tracking/soccer/robots/g1.py)
- 获取/来源：[上游位置](https://github.com/TeleHuman/HumanoidSoccer/blob/e72e470230047dedaf66df0983f1d0ab746faeb5/source/whole_body_tracking/soccer/assets/unitree_description/urdf/g1/main.urdf)
- 注意：原作者使用 UrdfFileCfg，首次加载需转换；保留 unitree_description 的 meshes/。

### 足球 USD

- 类型：repository
- 位置：`source/whole_body_tracking/soccer/assets/soccer/soccer.usda`
- 核查：配置/上游目录指向该位置；本轮未下载资产本体
- 定义：[源码](https://github.com/TeleHuman/HumanoidSoccer/blob/e72e470230047dedaf66df0983f1d0ab746faeb5/source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_flat_env_cfg.py)
- 获取/来源：[上游位置](https://github.com/TeleHuman/HumanoidSoccer/blob/e72e470230047dedaf66df0983f1d0ab746faeb5/source/whole_body_tracking/soccer/assets/soccer/soccer.usda)

### 踢球参考动作

- 类型：repository
- 位置：`motions/soccer-standard`
- 核查：配置/上游目录指向该位置；本轮未下载资产本体
- 定义：[源码](https://github.com/TeleHuman/HumanoidSoccer/blob/e72e470230047dedaf66df0983f1d0ab746faeb5/README.md)
- 获取/来源：[上游位置](https://github.com/TeleHuman/HumanoidSoccer/blob/e72e470230047dedaf66df0983f1d0ab746faeb5/motions/soccer-standard)
- 注意：已在固定源码目录列表核到 10 个 npz；本包未复制 npz。作者 ckp/policy_30000.onnx 是另一个可选复现模型。

## 原生 setting

移动球初速度 x/y ±0.3 m/s；继承每 1–3 s 身体推扰，平面速度 ±0.5 与角速度扰动。不是只看到“移动球”三个字就入选。

## 安装参考

安装 whole_body_tracking 扩展；资产 URDF 在线转换，另准备 motions/soccer-standard。本地只有源码/目录核查，未运行转换。

## 入口与控制边界

Tracking-Flat-G1-SoccerMoving-RNN-v0；--motion_path motions/soccer-standard；作者 ONNX 是复现参考，不能直接作为被测 agent 的动作工具。

## 动作接口

关节目标的全身策略接口；参考动作数据作为参考，不是直接提供作者训练好的控制器。

## 观测条件

保留原生观测定义；源码配置中可能包含 simulator state。原生 state 条件与 RGB/RGB-D 条件需分开记录，本包不擅自宣称纯视觉。

## 原始评估与失败

源码 kick_success 检测足球接触，未单独保证球朝目标飞出。应保留原接触指标，同时明确最终踢球结果和不跌倒，不能用接触数伪装任务成功率。

## 接入风险

不能把作者真机成功率直接当成该 sim 配置/该接触判据的成功率。推扰继承已查，资产/运动数据未 GPU 加载。

## 为什么需要新反馈

球的实际位置与身体被推后的姿态共同改变支撑腿和踢球脚所需动作。慢球本身可等，受扰全身支撑不能一直等。

## 反馈为何不能慢

失衡恢复与单脚支撑/触球有物理时限；若给现成稳定行走+踢球策略，agent 只做调度，就不再测试这层能力。

## 难度结论

全身支撑、外力和触球耦合，有较高研究价值；如果改成已有全身 policy 的高层命令接口，结论会改变。

## 已有视频/轨迹查看记录

实际抽查官网 “Kick moving soccer” 两段实机片段，约 0.08 s 与 3.08/3.44 s，看到来球和踢出阶段；未从视频反推运行频率。

## 源码定位

- [移动球配置](https://github.com/TeleHuman/HumanoidSoccer/blob/e72e470230047dedaf66df0983f1d0ab746faeb5/source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_flat_env_cfg.py#L393) · [包内文件](../../sources/TeleHuman/HumanoidSoccer/source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_flat_env_cfg.py)
- [继承的推扰](https://github.com/TeleHuman/HumanoidSoccer/blob/e72e470230047dedaf66df0983f1d0ab746faeb5/source/whole_body_tracking/soccer/tasks/tracking/tracking_env_cfg.py#L207) · [包内文件](../../sources/TeleHuman/HumanoidSoccer/source/whole_body_tracking/soccer/tasks/tracking/tracking_env_cfg.py)
- [接触成功指标](https://github.com/TeleHuman/HumanoidSoccer/blob/e72e470230047dedaf66df0983f1d0ab746faeb5/source/whole_body_tracking/soccer/tasks/tracking/mdp/kick_detection.py#L262) · [包内文件](../../sources/TeleHuman/HumanoidSoccer/source/whole_body_tracking/soccer/tasks/tracking/mdp/kick_detection.py)
- [作者实机演示](https://soccer-humanoid.github.io/)
