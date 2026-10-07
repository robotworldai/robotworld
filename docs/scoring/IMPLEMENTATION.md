# World 状态成功判定 v1

16 项任务、27 条规则已接入实际评测器，其余 68 题沿用已有整题判定。上游 checkout 与 Codex 源码保持原样；新增场地、命令、扰动、目标观测和评分位于 World 外部层。这是 World 任务版本，不宣称与官方原任务等价。

## 如何运行

正式模型评测仍由本地 Codex 驱动；各 `scripts/run_<benchmark>.sh` 默认 `SCORING_PROFILE=auto`，对这 16 项选择 `world-state-v1`，其他任务使用原评分。预算默认 `TASK=profile`。例如：

```bash
ROLLOUTS=3 bash scripts/run_go2_push.sh
SCORING_PROFILE=native ROLLOUTS=3 bash scripts/run_go2_push.sh
```

`native` 恢复原目标和原评分预算。两种版本不得混合统计；resume 会检查评分版本。低层 `native_project_launch`、WheeledLab Docker launcher 和旧 `scripts/eval/native_projects.py` 默认仍为 `native`，需要显式加 `--scoring-profile world-state-v1`。

无需模型的验证：

```bash
# 构造状态轨迹的成功、失败、缺字段和时序边界测试
python scripts/validate_world_success.py
# 每项默认8步，仅检查容器、真实状态提取和接线，不是做题评测
python scripts/validate_world_success.py --runtime
# 采用完整World考程；保留原生提前失败，不延长已失败的回合
python scripts/validate_world_success.py --runtime --full-duration --output var/validation/world-success-full
```

运行器会冻结本次使用的 `runner-environment/`，防止运行过程中修改主目录影响延迟导入。所有诊断目录均保留，不覆盖已有失败尝试。

## 判定与记录

`final` 检查结束状态；`hold` 检查末段连续保持；`always` 检查固定窗口。重置与每个控制步读取真实状态，用模拟时间计时。**不是每个物理子步采样**，不能保证控制步之间也一直满足。

- 原生失败或声明的场景越界优先，记失败。完整有效回合到时未达标也记失败。
- 未完成考程、资产/状态字段缺失、NaN、时钟不符、场景未按配置应用，不能记成功；有效性问题单列，`success=null`。
- 路线按有向门顺序更新，跳跃按真实腾空/抬升/落地计数，颠球与恢复复用原事件记录；模型声明不能改变结果。
- 接触阈值、坐标系、新增目标定位见 `world-scene.json`、`observation-metadata.json` 和当前英文 prompt。目标定位是明确新增的理想观测；私有判定状态不暴露给模型。
- `result.json` 保存 `world_evaluation`、选定评分的 `success`、`native_success`、原指标、评分版本及停止原因。
- `world-success.json` 保存逐规则结果、有效性、时间、失败原因；`events/world-success.jsonl` 保存每个控制步的评测状态，无图像。
- 原有完整模型轨迹、无图像轨迹、代码工作区和视频继续保存。参考控制诊断标记 `diagnostic_only`，不得计入模型统计。

ANYmal 当前原外层适配器每动作实际执行 5×0.005s，故 World 20s 对应 **800步 / 40Hz**。没有为了符合名义50Hz而修改原物理步进。SteadyTray 在计分前用原零残差动作建立真实向上托盘接触，最多0.5s，初始化单独记录；未建立则报错，不瞬移物体。

## 已验证到哪里

[诊断报告与逐题视频](../../reports/validation/world-success-v1/README.md)：261 个构造状态正反例通过，16/16 个 Docker 场景真实状态读取通过。完整运行已实际触发原生失败、到时未达标两类结果。

**构造状态正例能触发 success，不等于物理控制器已经完成任务。** 三次无人机载荷参考控制完整回合均未满足全部条件；没有放宽阈值。当前没有每题的物理成功轨迹，也没有新的 GPT 成功率。阈值可行性和难度校准仍需后续独立实验。

## 代码位置

- `environment/evaluation/world_success/profiles.py`：可执行规则与预算。
- `monitor.py`：通用时序判定；`readers.py`：各bench真实状态提取。
- `scenes.py`、`terrain.py`、`routes.py`、`visual_road.py`：外部场景配置及事件。
- `runtime.py`：运行挂接、结果与英文目标说明。
- `validation.py`、`environment/tests/test_world_success.py`：构造状态与集成契约测试。

设计数据和依据保留在 `success-proposals.json`；原评分和历史prompt保持可对照。整体按题等权统计口径见 [UNIFIED_SCORE.md](../UNIFIED_SCORE.md)，未覆盖全题集不能报告完整题集SR。
