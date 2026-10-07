# 科二风格的精密驾驶场景

2026-09-28 本轮复测使用 `robotworld-driving-precision-v2`：修复实线检查误受车体 root 高度影响的漏洞，改为安全包络的地面投影检查。尺寸、步数、成功阈值均保持不变。旧 `precision-lock.json` 和历史录像保持原样，新增 `precision-v2-lock.json` 冻结修复版本。

RobotWorld自定义任务，使用MuSHR小车和现有WheeledLab实验镜像；**不是国内驾考官方题库或官方评分复现**。单位均为该小车的实际米制尺寸。上游车辆/动作源码和Codex源码不修改；外部配置将原生`no_reverse`改为False，使负轮速可真实倒车。每题上限2000控制步，50Hz/40秒。

| 场景 | 场地及挑战 | 成功条件 |
|---|---|---|
| rw-twin-beam | 2.5m长、0.30m高的双梁；每梁95mm宽，轮胎约41mm宽，两侧边缘黄色禁压线各12mm。起点横向偏置，需要先对准再上桥 | 四轮完整通过桥；胎面不得触黄边、不得掉桥；终点中心误差≤14cm、方向≤5°、速度≤0.035m/s持续1s |
| rw-reverse-bay | 约1.22m深的横向通道，左右邻车和对面车辆限制调整；车位宽0.48m、深1.01m，黄色侧线/后线 | 真实倒车跨过白虚线入口，入位后的累计倒车距离≥0.45m；全安全包络入位、中心误差≤2.5cm、车头朝通道且误差≤3°，低速停稳1s；全程不压实线、不碰邻车 |
| rw-parallel-park | 前后夹车，车位长1.14m、宽0.54m；对面停放车辆和实线限制通道 | 倒车进入车位，入位后的累计倒车距离≥0.50m；全安全包络入位、中心误差≤2.5cm、与邻车平行且误差≤3°，低速停稳1s；全程不压实线、不碰邻车 |

停车题黄色实线是禁止边界，**接触就失败，不等到整车越过去才失败**。使用0.60×0.36m的声明安全包络与25mm宽的实线做投影相交检查，比只看车中心严格；邻车另加10mm安全间隔。白色虚线仅表示允许穿越的车位入口，不能将它也设成禁压线，否则无法入库。停稳速度上限0.035m/s。倒车入位、全车入位、位置、方向和停稳时间必须同时满足；模型说completed不算完成。

桥梁判据读取私有四轮link位置与声明胎面宽度，检查胎面是否侵入黄色边缘。这是可重算的几何安全判据，不冒称测到了真实接触力。梁、坡道、地面、车辆障碍均有PhysX碰撞。场地/邻车为程序生成的训练场几何，并非照片级街景。

## 模型看到什么

- 只有车载前视与后视RGB、轮速/转向编码器、IMU角速度与倾角。后视位于车体后方0.22m、高0.24m，向下约35°，不作镜像；前视向下约9°。
- 模型在工具边界看到有明确FRONT/REAR标签的640×360图片；`coding_control`每步读对应48×27 RGB8像素（`front_camera`、`rear_camera`字典中的`pixels`）。两路都固定在车辆上，随车移动。
- 不给地图、全局坐标、目标坐标、距线距离、泊车阶段、侧滑真值或自动规划路线。车位、绿色中心标记和边界只能由图像判断。
- 第三视角和场地鸟瞰图只供开发者复盘，不挂给模型。后台评分仍用真值，但评分文件不在agent沙盒中。
- 编码器/IMU暂为理想读数，无新增噪声；这是当前公开的传感器假设，不宣称等价真车感知。

## 文件与运行

本目录的`precision_specs.py`是私有场地尺寸和阈值，`precision_geometry.py`生成场地/车辆/实线，`precision_environment.py`使用上游已有开关启用倒车，`precision_scoring.py`独立判定。`precision_prompt.py`仅导出自然语言任务，绝不序列化私有spec。

在World根目录：

```bash
bash scripts/eval/wheeledlab_precision.sh list
bash scripts/eval/wheeledlab_precision.sh run --model gpt-6-astra --codex-home var/auth/robodojo-codex
# 单项
bash scripts/eval/wheeledlab.sh run --cases rw-reverse-bay --model YOUR_MODEL --codex-home /path/to/model-home
# 独立重算本轮冻结评分器
python third_party/benchmarks/wheeledlab/robotworld/precision_replay.py PATH_TO_RUN --check
```

保留前视`video/onboard.mp4`、后视`video/rear.mp4`和复盘`video/camera.mp4`，以及完整/no-images轨迹。三路均每控制步一帧，50fps，另加初始帧。

## 可解性与成绩边界

`precision_reference.py`和`--mode reference`仅用于**带真值的工程可解性诊断**：先搜索合法路径，再经原生轮速/转向接口执行，不能改位姿、跳步或放宽评分。此模式不会调用Codex，也绝不当作GPT成绩；这份代码/地图不在模型沙盒内，工具也不能调用它。

只找到几何路径不足以证明物理可解。正式难度判断需要同时查看工程控制器实际通过证据和源码Codex在传感器条件下的表现。当前验证产物在`var/runs/docker/wheeledlab/precision-v1/`，初期失败保留，不混作模型成绩。Isaac6.0.1/IsaacLab2.2仍为实验兼容配置，原MuSHR轮胎convexHull回退限制继续适用。

逐轮结果见[PRECISION_VALIDATION.md](PRECISION_VALIDATION.md)。
