# 任务与评测协议

| ID | World 任务 | 上游类 | 原生 STL |
|---|---|---|---|
|22|运动小球接取|FrankaBallCatching|always[50:299](XY距离 <=0.1m)|
|23|托盘小球平衡|FrankaBallBalancing|always[50:200](XY距离 <=0.25m)|
|24|插销插入|FrankaPegInHole|always[250:299](XY距离 <=0.1m)|
|34|接触异常后的取消与降级|同一原版 PegInHole + 外部记录器|自定义恢复判据；原版peg分数单列|

上游 `Evaluation/manipulator_eval.py` 默认每回合最多300个action，单环境，随机评估预算100回合；本套件默认先每题1回合、seed7。`VecEnv.reset` 本身执行一次零动作，初始观测也进入轨迹；不过上游trace=states随后copy_会把首样本覆盖为第一个动作后的观测，评分输入复现这个别名行为（原始native_trace.json仍保存真实reset）；原生 `progress_buf>=episodeLength-1` 会在之后298个策略action时终止，合计299个轨迹样本。启动器尊重终止，不为凑足300额外reset/step。trace索引从1开始，是样本索引，不是秒。

上游monitor+optimizer真实实现：22/23对RTAMT返回序列取min，24尽管公式是always，却在optimizer的特殊分支取max；成功为严格 `robustness > 0`。保持这些细节，不擅自修正。距离取剪裁后的native observation的XY分量，不检验3D插入深度；接取/托盘也不是接触或抓取认证。`dangerous_rate`为距离大于0.2m的比例（peg为0.37m）。RTAMT直接调用上游monitor，独立Python防止依赖冲突。距离提取保留float32的torch norm。RTAMT0.3.5公开工厂改名，通过外部别名适配旧类名。completion_time同原版，是满足条件的第一个0起始样本索引，非秒；原版无满足样本时mean(empty)=NaN，JSON里用null表示。22/23按原版range(len(trace)-5)检查连续5帧<=0.1m，24取首次<=0.1m。

初始条件沿用上游随机范围：22球xy偏移±0.05m，vx固定1、vy固定0；23球xy偏移±0.15m；24/34桌面xy偏移±0.1m。原生 `is_action_noise=True`，std0.5，发生在action剪裁之后。所有任务保留这些场景、噪声和物理配置。Isaac6升级本身单独标记实验兼容。

## 动作与工具

原生9维action：7臂关节+2指关节。工具 `move_joints` 暴露真正可控制的7维，末2维填0，因为上游会覆盖指关节目标。命名的arm状态数组只含7个旋转关节；手指位置/速度/目标另以米、米每秒报告，native observation仍保留上游9DOF排列。每步7个臂关节同时更新，delta=action×0.125rad，再受关节限位；不能当成绝对角或EEF位姿。2物理子步/控制步，配置physics dt=0.0083秒，Core整数频率换算后实际约1/120秒，控制dt约1/60秒；configuration.json记录实测dt。

- `observe`：不推进物理，读取当前状态和图像历史。
- `move_joints`：同时施加7维原生关节增量，1–50个控制步。
- `coding_control`：模型定义 `control(obs,memory)`，每控制步获得新反馈并返回7维action；代码受限解释器，不获得仿真句柄。最多300步，受剩余预算截断。代码、输入、输出和memory归档。
- `cancel_action`：仅ID34，在真实异常后取消旧段并锁定降级控制。

完整system/tool schema随run存入prompt.json。图像为当前+4history、interval2（工具边界）；视频每控制步记录，互不采样影响。

## ID34：contact-recovery-v1

只测插销与桌面的真实PhysX contact pair力，排除手指正常握持力。>=10N连续2物理子步视为异常。当前动作完成至控制步边界即中断剩余段，仿真暂停，把观测返回给模型；并不自动记取消/恢复成功。模型必须显式调用cancel_action，然后才能继续施加动作。

取消不模拟瞬间刹车、不改当前姿态、不改资产；只丢弃旧命令段并锁定未来增量上限。降级控制器在原生加噪声后，把每步臂关节target变化限制为±0.00625rad（正常最大请求的5%）；保持手指原逻辑。限制的是target变化，实际速度仍受惯性/碰撞影响，日志同时保存实测速度。

恢复判据：有真实触发 → 明确取消 → 至少10降级控制步，且结束前连续10控制步的每步峰值接触力<5N（之后再次高接触会撤销恢复状态）。`not_covered`无触发，`failed`触发未恢复，`recovered`满足；未覆盖success=null，不计恢复成功。原生peg结果与恢复结果分开，不用“tool执行完毕”代替终态验证。阈值为World自定义v1，非上游官方标准。未知/无效传感器读数导致错误，不能用0伪造无接触。
