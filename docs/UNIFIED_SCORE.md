# RobotWorld 整体成功率

当前活动题集：**84题、20个benchmark**。AI-CPS34接触异常退让已按用户要求移除。已有整题成功的68题沿用其原判据；另外16项采用用户认可的World状态式设计（27条状态检查），已实现为 `world-state-v1`，通过构造状态正反例及真实环境状态读取；物理可行性仍待校准。

## 主指标：各题等权

对题i的有效rollout，成功记1，失败记0：

`task_SR[i] = successes[i] / valid_rollouts[i]`

`overall_World_SR = mean(task_SR[i] for all 84 selected tasks)`

展示为百分数。先平均同一题的rollout，再对题等权平均；多跑某题不会提高该题权重。各题有效rollout数相同，才等价于全体成功次数除以全体有效次数。

例如两题分别1/2和9/10成功，按题等权整体SR为(50%+90%)/2=70%，不是10/12。若每题都跑10次，则宏平均与汇总次数比例相同。

各benchmark SR和能力类别SR可作分组结果另报。主整体SR不再将benchmark均值二次等权平均；否则只有1题的benchmark会得到与15题的benchmark相同权重。这是World题集的统一指标，不声称是上游官方统一评分。

## 单次结果与完整覆盖

- 原生任务失败、有效回合超时未完成：success=false，计入分母。
- 资产、模拟器、传感器或模型服务错误：invalid/error，success=null，单列原因和覆盖率；不能伪装成机器人任务失败。
- 未运行、中断、缺少已验证checker：pending/unscored，不能默认为成功或失败，也不能偷偷从全题平均中删除。
- 计划题数84；各题计划rollout数、有效数、成功数、invalid数均保存。按预先约定规则补齐所有题的计划有效rollout后，才发布可比较的全量分数；未齐只能报告明确子集的结果。
- 保留所有尝试及错误记录。不能只重跑模型失败、挑最好的一次；中断重试按既定resume逻辑从该题/rollout重启，逻辑rollout不可重复计分。

## 辅助指标与可比性

原reward、误差、BEHAVIOR Q、阶段分和子回合率另报，不能与0/1混合平均。逐题同时报告重复rollout的方差；只有1次无法估计样本方差。比较模型时固定题集版本、seed集合、环境/物理版本、预算、code_control开关和评分版本。

## 当前状态

场景层与checker已接入；**尚无基于这16项新checker的正式全套结果**。现有历史reward/子回合率不能在没有轨迹证据时反推成新的success。单题summary已按所选评分版本提取有效success并排除诊断回合。上述跨84题宏平均仍是统一报告协议，尚未自动生成完整覆盖的全套分数。

实现说明见[IMPLEMENTATION.md](scoring/IMPLEMENTATION.md)，诊断及视频见[验证报告](../reports/validation/world-success-v1/README.md)。

场景及逐条来源见[评分页面](../robotworld-scoring-guide.html)和[设计JSON](scoring/success-proposals.json)。旧的reward归一化综合分讨论已归档；本版主指标为所有题的World整体成功率。
