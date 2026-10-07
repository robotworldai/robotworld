# 接入验证记录

2026-09-27，本地 RTX 5090 / Isaac6.0.1 + IsaacLab2.2 实验兼容镜像。固定上游 commit 见 source.json。

## 已识别并外部处理

- 旧 IsaacLab 的默认资产根目录为 None：映射到本地、经 SHA256 校验的 NVIDIA 官方4.5地面资源。
- `ManagerBasedEnv.reset` 在 `assets_loading()` 循环中无法结束：probe-03 的 faulthandler 栈定位到 manager_based_env.py:321；关闭该等待开关，采用不推进物理的有限渲染预热。
- 早期 Replicator 相机预热长时间未完成，相关尝试被手动中断，未据此断言底层死锁。最终评审相机：改为 SimulationContext.render + 已有 render pump，逐控制步读取 RGB。不改变原生相机配置。
- probe-04 完成4步且保存结果后，Kit teardown 出现 busy TaskGroup 断言；官方 skip_cleanup 退出路径在后续 f1tenth-zero-03 验证通过（exit 0）。

## 已知上游资产/运行时限制

MuSHR USD 的轮胎 mesh collider 被 PhysX 报告回退为 convexHull。适配器没有修改 collider，但跨版本运行的接触行为仍不能宣称等价。F1TENTH USD 中有旧 ROS OmniGraph 节点，在本运行时出现未注册警告；本任务实际控制由上游 IsaacLab action manager 完成，不通过这些 ROS 图。地形 USD 的 color_121212.hdr 不在上游 checkout 内，未用其他图片替换。

自动化契约检查：13项通过（动作范围/非有限数值、工具屏蔽、观测隔离、原生评分同时失败、四项suite预算、共用模型配置）。实际运行记录随接入测试补充在下方。

## 实际运行

输出根目录 `World/var/runs/docker/wheeledlab/`。以下是链路检查，不能当作任务成功率评测。

| 运行目录 | 验证结果 |
|---|---|
| f1tenth-zero-03 | 原生零动作4步，视频5帧/50fps，exit0；已目视检查车辆和地面 |
| elevation-probe-02 | 原生零动作4步，原生高程/目标观测和5项终止判据可读，exit0 |
| visual-zero-01 | 原生零动作4步，视频5帧/5fps；40×80原生灰度图非空，policy向量3208维，exit0 |
| mushr-codex-smoke-01 | 本地源码Codex + gpt-6-astra，3次coding_control分别15/15/10步；累计40步，视频41帧/50fps，exit0 |
| mushr-native-horizon-01 | 原生零动作250步，time_out=True、truncated=True，精确5秒，exit0，无第二回合reset |

Codex 来源 `World/codex` commit `8ae55c863db26d417e83390c5854f1144114276b`；运行目录中的 agent-boundary.json 记录源码版本和实际二进制 SHA256。模型通过独立源码 app-server 调用，当前交互助手没有代为生成控制动作。Codex消息、工具请求/回复及控制回调逐步信息均在 events/ 和 events/no-images/。

visual-zero-01 发现初始 policy 图像在预热前计算，首帧偏暗；已将有限预热移到原生 reset 之前，随后由原生 reset 生成初始观测，避免把缓存的预热前图像给模型。后续复测见下方。

visual-codex-smoke-02：本地源码Codex + gpt-6-astra 已通过带原生图像的4步链路检查，4次coding_control各1步，5帧视频，exit0。预热前置后首帧仍偏暗，后续帧有黑白路径；因此不能据此宣称原生相机所有视角均已验证或与4.5渲染等价。本接入保留上游相机外参、黑白场景和图像增强，未额外改造为RGB驾驶任务。

## 优先漂移与地形补充验证

- `mushr-codex-native250-01`：源码Codex + gpt-6-astra，250步/5秒；time_out=True，out_of_bounds=False；251帧视频/50fps，exit0。原生累计reward=595.168790，side_slip加权积分约0.60789。这只说明该回合未越界并取得这些原生指标，不宣称完成连续漂移或产生二值成功率。一次控制程序缩进错误作为工具错误返回，模型随后修正，完整过程保留在events。
- `elevation-video-01`：真实原版 `huge_compact.usd` 高差地形，已目视检查坡面/凸起/障碍；zero动作检查在第7步原生stuck终止（计划10步），没有隐藏该失败，8帧/10fps，exit0。该轮是场景/视频/终止链路检查，不是GPT任务评测。初始截图 `video/initial.png`。
- `validation-summary.json` 位于上述运行根目录，列出8轮最终验证；ffprobe核对视频帧数，逐项奖励累计与原生总奖励一致（浮点容差内）。各个上游checkout和本地Codex保持clean。
