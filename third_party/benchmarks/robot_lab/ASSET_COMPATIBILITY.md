# T11 A1 脚部兼容修复

## 2026-09-28：关节摩擦迁移修复

后续单变量对照定位到第二个独立问题：上游 A1 `DCMotorCfg(friction=0.0)` 在 4.5 实际读回为 0，但 IsaacLab 2.2 + 6.0.1 仍留下 USD 的旧式 `jointFriction=0.2`；新三分量摩擦接口同时读回全零，不能据此认为旧参数已经清零。

外部 `environment/benchmarks/robot_lab/friction_compat.py` 给 T11 的 Articulation 加入初始化检查，按上游明确的零摩擦要求同步写入并读回两套 API。只允许此零摩擦契约，不把旧无量纲系数照搬成新接口的摩擦力矩；非零新式参数或读回失败立即报错。`a1-friction-compatibility.json` 保存修复前后实际值。

生产入口在 6.0.1 下默认启用。`WORLD_A1_FRICTION_COMPAT=0` 仅供复现历史问题；诊断脚本为保留原对照默认关闭，验证修复需加 `--friction-compat`。原始源码、资产、奖励、终止、动作、PD 参数不改。

同 USD / 同 Lab 的 500 步保持测试，根位置跨版本最大差异由 12.44 cm 降至约 0.23 mm。旧 GPT 动作在修复后的原生条件仍触发接触终止（第 50 步），不能将兼容修复等同于任务成功。完整证据在 `var/runs/docker/robot_lab/physics-friction-fix-report.md`；早先 `a1-feet-gpt01` 是修复前的历史轨迹。

## 问题与依据

固定 AgileX 提交中的 URDF 已将四个脚部固定关节标为 `dont_collapse="true"`，但其打包 USD 只有 13 个刚体，原奖励需要的 `R.*_foot` 不存在。

上游 [robot_lab v2.2.1](https://github.com/fan-ziqi/robot_lab/releases/tag/v2.2.1) 有针对 A1/B2 脚部添加该标记的修复。[Issue #88](https://github.com/fan-ziqi/robot_lab/issues/88) 的 Go2 用户也报告类似缺脚部问题，部分用户通过官方 USD 绕过转换解决；不能把 Go2 方案直接当作已验证的 A1 资产。

本机 Isaac Sim 6.0.1 已用 `URDFImporter` 替代 IsaacLab 2.2 所调用的旧 `_urdf` 接口。新导入器的 `merge_fixed_joints` 遍历所有 fixed joint，不判断 `dont_collapse`。

## 外部实现

代码：`environment/benchmarks/robot_lab/asset_compat.py`，运行配置：`runtime-profiles/a1-feet.json`。

1. 读取固定版本原 URDF 和 mesh，不使用新下载版本替换；所有生成物写到每轮输出的 `a1-reimport/`。
2. XML 预处理期间暂存四个带保留标记的 foot joint，调用 Isaac6 自带合并函数处理其它固定关节，随后恢复四个 joint，再交给导入器；保留脚部原质量、惯量、碰撞几何和固定约束。
3. 原 URDF 重复定义 `grey`，新解析器拒绝重复定义。在派生 XML 中保留旧 USD 使用的第一个灰色 `(0.2,0.2,0.2)`，移除重复的橙色定义；不改原文件。
4. 调用 Isaac6 原生 URDFImporter，关闭二次固定关节合并。将其嵌套刚体路径整理为原配置要求的 `Robot/base`、`Robot/<link>`；保存世界变换并用 USD NamespaceEditor 更新关节引用。
5. 检查总质量和活动关节参数，断言 12 个活动关节及四个脚部刚体存在。原任务奖励、终止条件、事件、actor 观测、动作尺度和 500 步上限不变。

该配置重用了已有镜像，无需下载新镜像或改 Docker 内的上游源码。它是**派生资产的实验兼容版本，不宣称与官方原运行时物理数值等价**。原资产和 Codex 源码不改，原 USD 路径仍可用 `--runtime-profile default` 选择。

## 运行

在 World 根目录：

```bash
# 默认已使用 a1-feet；探针不调用模型
bash scripts/eval/robot_lab.sh probe T11

# 本地源码 Codex + 本地 Docker，原生最多 500 步
bash scripts/eval/robot_lab.sh run T11 --codex-home /path/to/codex-home

# 直接 Python 入口需显式选兼容配置
python third_party/benchmarks/robot_lab/docker/run.py \
  --task T11 --mode probe --steps 4 --runtime-profile a1-feet \
  --output /absolute/path/to/new-run
```

## 验证证据

- 成功探针：`var/runs/docker/robot_lab/a1-feet-probe04/`。`exit.json` 为 `infrastructure_ok=true`；4 个动作步、dt=0.02s，原奖励成功返回，没有提前终止。
- `independent-asset-audit.json`：17 个刚体；12 个活动关节 + 4 个脚部固定关节；所有关节引用可解析；路径整理前后世界位姿最大误差 0；总质量约 13.741 kg，四个脚部各 0.06 kg。
- 每轮 `a1-asset-compatibility.json` 保存原 URDF SHA256、生成资产 SHA256、质量、刚体/关节/碰撞体列表；`a1-reimport/urdf-audit.json` 保存预处理质量及活动关节参数检查。
- CPU 回归检查：`test_robot_lab_asset_compat.py` 验证原文件不变、保留标记恢复、拒绝意外质量/关节限位变化、profile 不替换任务定义。
- 模型验证输出：`var/runs/docker/robot_lab/a1-feet-gpt01/`，`exit.json` 为 `infrastructure_ok=true`。本地源码 Codex + gpt-6-astra 在 101 步（2.02 秒）触发原 `illegal_contact`；apply_action 7 次、coding_control 2 次，没有完成倒立。原任务无二值 SR，记录 success=null。视频 `video/camera.mp4` 经 ffprobe 确认 50 FPS、102 帧（含初始帧）；完整及 no-images 的 codex/environment/tools 事件分别为 78/146/18 条，逐类一致。

此前 probe01–03 保留诊断记录，不能当作有效评测回合。原始资产搜集记录在 `asset-research/2026-09-28/README.md`，它记录的是修复前状态。
