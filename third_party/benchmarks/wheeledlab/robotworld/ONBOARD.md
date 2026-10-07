# 车载传感器观测边界：robotworld-onboard-v2

2026-09-28按用户要求，将四个RobotWorld自定义驾驶场景从“精确状态+完整地图”切换为实际车辆可搭载的传感器类型。原生WheeledLab四项配置不受此修改影响；自定义场景的原有成功/失败评分不变。

## 提供给模型与控制代码

| 内容 | 模型工具回复 | coding_control 每控制步 |
|---|---|---|
| 车载前视RGB | 640×360图像，当前+4历史（interval2工具观测） | 同一图像的48×27 RGB8像素，展平数组3888值 |
| 轮速编码器 | 每个轮关节的rad/s | 同左 |
| 转向编码器 | 左右前轮转向关节rad | 同左 |
| IMU类读数 | 车体角速度rad/s、roll/pitch rad | 同左 |
| 执行状态 | 控制步号、dt、上次请求动作、回合结束标记、剩余预算 | 同左，额外program_step |

传感器目前是**理想模拟值，没有增加噪声/偏置/丢帧**。IMU倾角不是全局yaw；编码器轮速不等于打滑时的真实地速。没有GPS、激光雷达、深度图或视觉语义检测器。模型可以自行从合法观测估计里程、路沿和障碍，接口不提供估计结果。

相机固定在车体前方0.22m、高0.24m，约向下9°、水平FOV约82°，跟随车体姿态。图像仅来自真实渲染，不读取语义分割或评分器标签。控制器像素：`obs['front_camera']['pixels'][(y*48+x)*3+c]`，c=0/1/2对应RGB。相机位置变换会在模拟器内部用到机器人pose以实现安装效果，该pose不导出。

## 明确移除

- `native_policy_terms`：原生BlindObs含有模拟器world pose与真实body velocity，不能直接透传给这四题。
- `vehicle.position/yaw/linear_velocity/body_velocity`、`navigation`、`traffic_gate`。
- 下一检查点、检查点索引/数量/进度、路线预览、世界坐标终点及目标yaw。
- 闸门精确位置、相位/周期、净空数值和是否能通行的布尔结论。
- prompt内完整地图、路径坐标、障碍物坐标、摩擦分区和扰动参数。

任务prompt只给车辆/动作/相机参数、目的地外观和自然语言规则，以及公开的成功阈值。有限道路任务沿路上的蓝色标记前进，终点是绿色停车标记；漂移任务以绿色标记为起终点。闸门题新增可见的红色停车框，使必须停车的位置能由相机观察到；只有视觉标记，没有改变碰撞、停车区坐标或评分阈值。

## 实施边界

观测通过白名单重新构造，不从原生dict删除几个字段后继续透传。模型与程序各自拿到同一传感器白名单；程序额外获得低分辨率图像，不能访问Python simulator句柄。Shell仅能读agent的允许观测和工作目录。完整World、scenario.json、独立评分轨迹均未挂载进Codex沙盒。

`events/scoring.jsonl`仍由开发者用来评审，后台可使用真值判断成功，但不能被agent读取。旧评分协议`robotworld-wheeled-v1`和新观测协议是两个分别记录的版本。每轮保存`observation-profile.json`、环境/观测/工具源码快照和哈希；旧结果不会被重标为新观测成绩。

视频有两路：`video/onboard.mp4`是实际车载输入，`video/camera.mp4`是仅供复盘的第三视角。二者每控制步一帧、50fps；单独记录初始帧。低分辨率逐步原图在`sensor-frames/front-lowres/`。完整events保留程序拿到的像素；no-images版本移除像素并保留形状/步号/哈希。

## 验证

- 27项测试通过，包括原生接入回归、自定义评分、公共/程序两通道防真值泄露、prompt不序列化私有spec、控制代码可读像素而不可读world pose，以及no-images导出。
- `var/runs/docker/wheeledlab/onboard-v2/gate-sensors-smoke-01`：8步Docker实测，exit0，前视图能看到道路/闸门/停车框；两路视频均9帧/50fps，最终观测仅含上述白名单。
- `var/runs/docker/wheeledlab/onboard-v2/courtyard-codex-smoke-01`：本地源码Codex + GPT-6完成40步链路测试，exit0。模型自行写了从RGB像素识别蓝色路面标记的控制程序；首次误读像素字典后自行纠正。逐步回调核查未含被禁止的真值字段，两路视频各41帧/50fps，独立评分重算一致。此轮仅0.8秒仿真，不是完成任务的证明。随后prompt补充了完整像素访问表达式，避免结构歧义。
- `var/runs/docker/wheeledlab/onboard-v2/courtyard-gpt6-2000steps-01`：2000步预算下，GPT-6在739步内完成园区配送。4次coding_control，识别蓝色路面标记后识别绿色停车区；6/6检查点，终点中心误差0.352m、方向误差11.19°、速度0.0137m/s，满足旧园区协议的0.65m/25°/0.12m/s并保持0.8s。两路视频各740帧/50fps，独立评分重算通过。其他三题在新版条件下尚未完成评测。

后续precision题额外提供车载后视（向下35°，不镜像）并启用原生倒车开关；前后图像明确标注，仍不暴露世界位姿/地图。参见[PRECISION.md](PRECISION.md)。

运行仍用 `bash scripts/eval/wheeledlab_custom.sh run --model YOUR_MODEL --codex-home /path/to/model-home`，不需要另建镜像，镜像挂载World外部接入代码。原始WheeledLab checkout和Codex源码未改。
