# T04 · 绳牵横梁上的小球抗扰定位

来源：[Wenminggong/aerial_balance_bench](https://github.com/Wenminggong/aerial_balance_bench)

固定 commit：`d5bf6eae9f955bc03609d9adaedcc9885b31d6b9`

原 task/config：`environments/configs/template_eval_disturbance.yaml`

运行依赖：Isaac Sim 4.2.0 + Isaac Lab 1.4.0 + Python 3.10（作者 tested stack）

接入状态：本包未实现或验证 RoboWorld_Scaffold adapter；提供原生任务接入材料。

## 资产与获取

### 无人机—绳—横梁—滚动物块系统

- 类型：repository
- 位置：`asserts/drone_rope_plank_horizontal_slide_block.usd`
- 核查：配置/上游目录指向该位置；本轮未下载资产本体
- 定义：[源码](https://github.com/Wenminggong/aerial_balance_bench/blob/d5bf6eae9f955bc03609d9adaedcc9885b31d6b9/environments/assets/drone_rope_plank.py)
- 获取/来源：[上游位置](https://github.com/Wenminggong/aerial_balance_bench/blob/d5bf6eae9f955bc03609d9adaedcc9885b31d6b9/asserts/drone_rope_plank_horizontal_slide_block.usd)
- 注意：上游目录拼写就是 asserts。需保留该 USD 的层级/关节结构；不能用任意 drone+球示例替换。

## 原生 setting

选作者已经公开的 external_disturbance_enabled=true 评测 YAML。无人机受限于竖直运动，经绳子改变横梁角度；扰动作用于横梁另一端。不是一般飞行悬停。

## 安装参考

按 README 安装 conda_env.yml；在配置好的 Isaac Lab Python 中运行。先用 zero_action_policy_eval.py 核查接口。

## 入口与控制边界

不是 Gym ID：使用 environments/configs/template_eval_disturbance.yaml。默认 delay-free 配置不是本清单选择的题。

## 动作接口

原 velocity 接口是竖直速度增量；提供球位置/速度、梁角/角速度等 11 维状态。原策略步长约 1/60 s。原 YAML 同时有 15 步动作延迟，必须如实记录。

## 观测条件

原 11 维状态：球位置/速度/加速度、梁角度/角速度/角加速度、drone z 位移/速度/加速度、目标与上一动作。是 state-based 设置。

## 原始评估与失败

原评测报告目标带误差、收敛时间、稳态误差与 success_rate；本设置目标带 0.01 m、回合 10 s。代码 SR 统计未显式排除所有物理提前终止，也没有单独要求在带内至少保持一整秒，不能把它描述成更严格的验收。

## 接入风险

仅选 template_eval_disturbance.yaml。源码 evaluator 的目标区判断可能接受末尾单帧，需核对失败标记与持续保持口径；README 的描述不能代替实现。修改评分必须另记为 RobotWorld 评分。

## 为什么需要新反馈

每个物理步加入新的 OU 随机扰动，使横梁另一端上下运动。实际梁角、球速和球位置决定下一次控制；未来扰动无法在起点编成确定的动作表。

## 反馈为何不能慢

梁的倾斜会持续加速小球，偏差不是等一等就自动消失。纠正太晚会离开目标带甚至越过横梁可行范围，触发失败。

## 难度结论

OU 扰动持续作用于横梁另一端，有直接的运行中新信息；但一个经典控制器可能就能解决，不能预先称圣杯难题。

## 已有视频/轨迹查看记录

检查源码、官方说明及评测器；没有实际观看该配置的成功/失败视频。

## 源码定位

- [作者公开的扰动评测设置](https://github.com/Wenminggong/aerial_balance_bench/blob/d5bf6eae9f955bc03609d9adaedcc9885b31d6b9/environments/configs/template_eval_disturbance.yaml#L1) · [包内文件](../../sources/Wenminggong/aerial_balance_bench/environments/configs/template_eval_disturbance.yaml)
- [每物理步新采样扰动](https://github.com/Wenminggong/aerial_balance_bench/blob/d5bf6eae9f955bc03609d9adaedcc9885b31d6b9/environments/robustness/robustness_manager.py#L137) · [包内文件](../../sources/Wenminggong/aerial_balance_bench/environments/robustness/robustness_manager.py)
- [原评分的实际逻辑](https://github.com/Wenminggong/aerial_balance_bench/blob/d5bf6eae9f955bc03609d9adaedcc9885b31d6b9/environments/evaluation/target_position_evaluator.py#L72) · [包内文件](../../sources/Wenminggong/aerial_balance_bench/environments/evaluation/target_position_evaluator.py)
