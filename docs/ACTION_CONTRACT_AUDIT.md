# 动作语义与英文 prompt 审计

本次覆盖评分清单中的 **20 个 benchmark、85 道选题**。同一 bench 共享控制器的题复用一份动作说明；RoboLab 按三种控制模式区分，踢球按 direct/hybrid 区分，小车按是否允许倒车区分。模型收到的 system/developer instructions、工具说明及新增示例使用英文；本文件是给维护者看的中文解释。

这次核对的是**接口定义和 prompt 实际接线**，不是85道题重新做完，也不是证明模型已经学会控制。没有改上游源码、动作数值映射、物理参数、成功判据、预算或已有运行记录。新说明在下一次创建 Codex 会话时生效，旧 `prompt.json` 是当时的真实快照，不回写。

## 每份说明必须回答的问题

1. 哪个输入对应哪个关节/控制通道；完整向量的顺序是什么。
2. 输入是绝对目标、相对实测状态的增量、相对上一目标的增量、相对默认姿态的偏移，还是速度/推力。
3. 单位、缩放、零点、输入限幅、目标限幅与执行器限制分别是什么。
4. 参考系是世界、机器人根、本体还是关节局部轴；姿态是WXYZ、XYZW、旋转向量还是特定欧拉参数。
5. `steps` 是保持同一目标，还是重复累加；控制步的模拟时间是什么。
6. 未给出的字段保持实测值、上次命令还是归零；普通工具和反馈代码是否不同。
7. 数值例子改变的是哪个**目标量**；不能将它说成一定发生的实际位移。
8. 多通道是同时还是先后执行；夹爪是在运动中闭合还是到达后闭合。

## 重点修正和补充

- **RoboDojo**：绝对世界坐标、夹爪中间抓取点，不是法兰；`pitch`绕世界X、`roll`绕世界Y，单位为度。这不是通常的RPY命名。例：z从0.90改成0.92只请求上移2cm；再次发送0.92不会继续累加。手臂先到达，随后执行同一调用中的夹爪变化。
- **BEHAVIOR**：末端为机器人根坐标绝对值，XYZW；底盘为本体速度m/s与rad/s；躯干为关节弧度，不是升高几厘米。普通调用省略末端/躯干字段锁定调用开始的实测值，反馈程序省略时保持上次程序目标。补充了实际躯干关节名称的顺序与限位，来源为机器人控制布局。
- **RoboCasa**：末端归一化增量按0.05m、0.5rad缩放；旋转是旋转向量。mode0相对实测姿态，mode1相对上次期望姿态，不能笼统认为重复10次一定移动10倍距离。躯干为向上的**滑动关节**，每步目标是实测高度加0.05×输入米。底盘原版还先按当前与初始yaw之差做特定符号的变换，再通过执行器范围将平移/转向缩放为1m/s、1.5rad/s每单位；已写出原公式及theta=0、pi/2的例子。Docker镜像中的5个相关控制源码/模型文件与本机源码SHA256一致。
- **RoboLab**：三种mode分别解释；绝对IK控制`ee_pos/ee_quat`对应的base_link法兰，不可拿另一报告坐标系`eef_pos/eef_quat`直接代替。relative_ik的0.5缩放和每步重新基于实测位姿的行为单独解释；四元数为WXYZ。运行镜像使用保留原约定的IsaacLab2.2源码层，不能从另一份IsaacLab6的文档推断四元数顺序。
- **AI-CPS**：七关节输入每步向**上一目标**加0.125×输入rad，与RoboCasa相对实测状态的增量不同。加入10步累加的例子、原动作噪声、零输入并不刹车、ID34取消后的限幅说明。
- **WheeledLab**：方向/轮速目标保持，不累加；前轮目标是tan(0.488×输入)rad。只读加载MuSHR与F1Tenth USD核对转向轴，正转向为本体+Z，正向行驶名义上左转，倒车yaw响应相反。保留轮胎侧滑的影响。所有模型侧任务标题已使用英文，报表中文标题保留。
- **Native项目与Bench2Dex**：已有运行时通道表保留，并同步嵌入`apply_action`和`coding_control`工具说明。表按真实解析后的关节顺序输出offset、scale、输入范围、处理后限幅，以及u=0、0.1、1对应的目标；增加从绝对角度反求u的例子，强调观测可能先做了默认姿态减法/缩放，不能直接复制为action。
- **HumanoidSoccer**：direct是绝对关节rad；hybrid是每个新原策略建议上的rad偏移，不逐步积分。对应运行模式的同一英文说明也进入工具定义。

## 全部 benchmark 覆盖表

| Benchmark | 选题数 | 动作语义与关键参考 | 实际模型入口 |
|---|---:|---|---|
| RoboDojo | 15 | 世界抓取点绝对m；特殊pitch/roll/yaw度；0关1开 | [adapter](../environment/benchmarks/robodojo/adapter.py)、[EEF执行器](../environment/benchmarks/robodojo/eef_executor.py) |
| BEHAVIOR-1K | 10 | 根坐标绝对EEF m/XYZW；本体速度；躯干绝对rad | [policy](../environment/benchmarks/behavior_1k/policy.py)、[control](../environment/benchmarks/behavior_1k/control.py)、[代码闭环](../environment/benchmarks/behavior_1k/code_control.py) |
| RoboCasa | 10 | OSC增量；底盘归一化速度；躯干实测位置增量 | [policy](../environment/benchmarks/robocasa/policy.py)、[control](../environment/benchmarks/robocasa/control.py) |
| RoboLab | 10 | 绝对关节/绝对IK/相对IK三种模式；根坐标WXYZ | [policy](../environment/benchmarks/robolab/policy.py)、[control](../environment/benchmarks/robolab/control.py) |
| HumanoidSoccer | 1 | direct绝对rad / hybrid相对新策略目标的rad偏移 | [policy](../environment/benchmarks/humanoid_soccer/policy.py)、[运行时说明](../environment/benchmarks/humanoid_soccer/control.py) |
| AI-CPS | 4 | 上一关节目标+0.125u rad，每步累加，保留噪声 | [policy](../environment/benchmarks/ai_cps/policy.py)、[control](../environment/benchmarks/ai_cps/control.py) |
| WheeledLab | 11 | 3u m/s目标；前轮tan(0.488u)rad；按题开启倒车 | [policy](../environment/benchmarks/wheeledlab/policy.py)、[guide](../environment/benchmarks/wheeledlab/control.py) |
| Bench2Dex | 9 | 全部active关节绝对rad，scale1/offset0；mimic保持原版 | [project](../environment/benchmarks/bench2dex/project.py)、Native共享入口 |
| SteadyTray | 1 | 默认关节姿态+运行时逐关节scale×u，不是累加 | [project](../environment/benchmarks/steadytray/project.py)、Native共享入口 |
| TTRL | 1 | 默认姿态+0.25×clip(u,-100,100)rad，21通道以运行时为准 | [project](../environment/benchmarks/ttrl/project.py)、Native共享入口 |
| ReflexBench | 1 | 根坐标绝对EEF m/WXYZ与二值夹爪；不能全零hold | [project](../environment/benchmarks/reflexbench/project.py)、Native共享入口 |
| Wheel-Legged-Lab | 2 | 六通道虚拟腿角/腿长/轮速目标；不是六关节角 | [project](../environment/benchmarks/wheel_legged/project.py)、Native共享入口 |
| Wheeled Quadruped | 1 | 原action manager的关节位置与轮速通道，分别rad与rad/s | [project](../environment/benchmarks/wheeled_quadruped/project.py)、Native共享入口 |
| Go2 Push | 1 | 默认关节姿态+0.25u rad；原PD | [project](../environment/benchmarks/go2_push/project.py)、Native共享入口 |
| robot_lab / A1倒立 | 1 | 默认姿态+原运行时scale×u rad | [project](../environment/benchmarks/robot_lab/project.py)、Native共享入口 |
| Digit | 1 | 原运行时关节顺序与offset/scale生成目标rad | [project](../environment/benchmarks/digit/project.py)、Native共享入口 |
| OmniIsaacGymEnvs / ANYmal | 1 | 默认姿态+0.5u rad；原PD80/2、力矩±80Nm | [project](../environment/benchmarks/omniisaacgymenvs/project.py)、Native共享入口 |
| Flamingo | 1 | 六位置目标+两轮速；腿部为电机侧角，齿比-1.5 | [project](../environment/benchmarks/flamingo/project.py)、Native共享入口 |
| OmniDrones | 2 | 四旋翼推力映射(u+1)/2；零为50%稳态推力而非悬停 | [project](../environment/benchmarks/omnidrones/project.py)、Native共享入口 |
| VolleyBots | 2 | Iris四原始旋翼输入，保留电机延迟；无自动悬停 | [project](../environment/benchmarks/volleybots/project.py)、Native共享入口 |

Native共享入口：[实际会话prompt](../environment/benchmarks/native_project/policy.py)、[运行时英文动作表](../environment/benchmarks/native_project/action_prompt.py)、[工具schema](../environment/benchmarks/native_project/control.py)。旧T编号仅用于内部任务映射，没有新增已删除的T04。

## 代码与导出

- [共享英文动作说明](../environment/benchmarks/action_contracts.py)：固定接口的system与tool复用同一文本；只有工具说明发生改变，不改变输入schema及执行逻辑。
- [英文导出目录](action-contracts/)：可直接阅读模型侧新增说明。Native示例使用已有运行的公开action metadata重新生成，不启动GPT/仿真，不重写历史prompt。
- [导出脚本](../scripts/export_action_contracts.py)：运行 `python scripts/export_action_contracts.py`。
- [数值与接线验证](../environment/tests/test_action_contracts.py)：执行固定源码中的纯计算方法，核查RoboCasa底盘/躯干例子；验证RoboDojo特殊角度和抓取点往返；检查实际会话和工具采用同份contract，检查模型文本为英文。

## 控制依据

- [RoboDojo姿态变换](../environment/robots/arx_x5/pose.py)。
- [BEHAVIOR观测与控制布局](../environment/integrations/behavior_eval.py)。
- RoboCasa：[控制配置](../third_party/dependencies/robosuite/checkout/robosuite/controllers/config/robots/default_pandaomron.json)、[OSC](../third_party/dependencies/robosuite/checkout/robosuite/controllers/parts/arm/osc.py)、[底盘](../third_party/dependencies/robosuite/checkout/robosuite/controllers/parts/mobile_base/joint_vel.py)、[躯干控制器](../third_party/dependencies/robosuite/checkout/robosuite/controllers/parts/generic/joint_pos.py)、[Omron关节与执行器定义](../third_party/dependencies/robosuite/checkout/robosuite/models/assets/bases/omron_mobile_base.xml)、[Gym动作转换](../third_party/benchmarks/robocasa/checkout/robocasa/wrappers/gym_wrapper.py)。
- RoboLab：[IsaacLab2.2差分IK](../third_party/dependencies/isaaclab22/checkout/source/isaaclab/isaaclab/controllers/differential_ik.py)、[apply_delta_pose](../third_party/dependencies/isaaclab22/checkout/source/isaaclab/isaaclab/utils/math.py)、[保留2.2层的镜像](../third_party/benchmarks/robolab/docker/Dockerfile.isaac601)。
- WheeledLab：[归一化处理](../third_party/benchmarks/wheeledlab/checkout/source/wheeledlab/wheeledlab/envs/mdp/actions/ackermann_actions.py)、[实际RC tan转向](../third_party/benchmarks/wheeledlab/checkout/source/wheeledlab/wheeledlab/envs/mdp/actions/rc_car_actions.py)。
- [VMC虚拟腿几何](../third_party/benchmarks/wheel_legged/checkout/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/mdp/actions.py)。

目标变化与物理结果仍须分开：相同控制目标在接触、侧滑、噪声、执行器延迟、不同兼容运行时下可能产生不同轨迹。说明完整是公平评测的前提，不是性能或物理等价的证明；本次也未据此宣称Flamingo的历史运行故障已经解决。

## 本次验证结果

相关接口、动作拼装、prompt接线、语言及数值回归共 **127 项测试通过**。数值检查包括实际固定版RoboCasa控制器的纯计算方法；小车USD轴核对没有启动物理场景。[验证记录](action-contracts/validation.json)保存修改文件哈希。没有重跑GPT任务，也没有把这项检查表述为85道题的物理实测。
