# 实施路线

0. 当前：目录与 Markdown 完成；上游代码未改，未下载或构建。
1. 核实 Codex 本地源码构建和动态工具文字/图像闭环；确认工具集合、指令覆盖、模型兼容、请求计数和结束控制。
2. 实现 runtime 及 fake benchmark，验证多次工具调用、串行执行、重复请求去重、错误、超时和资源清理。
3. 从 RoboProbe 迁移 EEF 执行器、pose、机器人说明；固定输入下比较新旧轨迹、夹爪顺序和反馈。
4. 验证 RoboDojo 无修改加载路径；通过官方 API 完成单 episode、官方结果、录像与日志。不可行时记录阻塞。
5. 完成 RoboDojo 数据清单、隔离镜像、固定场景 smoke 和小规模对照；运行清单锁定全部版本。
6. 按相同 checklist 接入 RoboLab、Behavior-1K、RoboCasa。每个 bench 独立完成仓库确认、许可/资产、生命周期、控制语义和官方评分核验。

## 验收

- Codex 二进制确实由 World/codex 指定 commit 构建；无全局安装 fallback。
- 所有上游工作区在执行前后保持一致；正式评测上游来源干净且可追溯。
- move_eef 实际驱动仿真并回传真实新图像，非仅 mock。
- 官方结束后不再执行动作；不将 assistant final 当官方成功。
- 对照 RoboProbe 的动作语义、轨迹与夹爪时序一致；agent 历史和预算差异单独声明。
- 运行结果可以从源码、配置、镜像、数据、seed 重建；数值非确定性另行记录。
