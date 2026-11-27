# 接入与验收

## 必须保留的信息

每次运行记录：上游 commit、精确 task ID/config、Isaac Sim/Lab 版本、机器人资产路径、seed、reset 范围、运行中事件、动作层级、观测内容、物理 dt、控制 decimation、episode 时长、模型/程序等待时环境是否继续前进。

1. **资产与原题可行性**：确认 USD/URDF 及其引用能加载；用作者控制器/基线验证同一配置可完成。原训练 checkpoint 是可行性参考，不自动给被测 agent 使用。
2. **观测条件**：原生 simulator state 与传感器条件分别标记。T03 的预测拦截量、T02 的预测器等要显式说明。
3. **动作边界**：电机 PD/IK/VMC 可以是公共执行层，但已完成平衡、接球、行走的任务 policy 会把被测能力移出 agent。先记录清楚保留了什么，不根据“eef”名称判断。
4. **时间语义**：若等待 LLM 时停止 sim.step，则快闭环困难被改变。基线与程序方案需要相同物理时间语义，并记录等待时执行的动作/旧程序。
5. **评分**：保留原 reward、failure、termination、timeout 与 episode 轨迹，不能把 reward>0、跑满时间或单次接触直接当成成功。对没有二值 SR 的题，先报原指标；新增 SR 必须另标 RobotWorld-defined，不能说是原题现成判据。
6. **简单解排查**：同一隐藏 seed 集比较保持动作、固定动作轨迹、慢反馈、快速反馈程序。允许真正能完成任务的简单解；若它们高分，则降级题目，不人为改分数证明 code 必胜。

## 特别容易接错

- T01：Stage 3 residual teacher 与 Stage 4 student 是不同 runner；物体 track_only 失败不能用 done 漏掉。
- T02：只触球和有效回球分开；预测器是否提供须锁定。
- T03：虚拟捕获区/杯子随 EEF 更新是原建模假设，不能说完整自然抓握。
- T04：固定使用 disturbance YAML；核对 evaluator 是否把末帧进入目标区当成持续成功。
- T05：对手 checkpoint、原始转子或 CTBR 接口分别固定，不混用。
- T06：Play 关闭 reward/termination；速度与漂移要求不能丢，否则停车也算存活。
- T07/T08：共用机器人资产，两个不同需求；保留推扰，Reactive 不等于 Perceptive。
- T09：Balance-Play 会移除 push；本包选择非 Play。
- T10：随机课程初始 30 N，与固定 120 N scheduled demo 不同。
- T11：back=抬后腿；需要抬脚、姿态指标，不只“不倒”。
- T12：原 kick_success 是接触检测，不能据此宣称踢到目标。
- T13：是双手目标跟踪，没有货物；14 s episode 不保证触发 10–15 s 的每次 push。
- T14：旧 OIG 任务要匹配 Sim 4.0 资产根。
- T15：TrackJump 的 Play 推扰幅度为零；资产必须 rev01_5_2。
- T16/T17：两题原模型都是 Hummingbird；action_transform=null 不要误接已有完整控制 policy。

以上是交接与验证要求，没有替你修改任务、实现 adapter 或规定一个新的成功阈值。
