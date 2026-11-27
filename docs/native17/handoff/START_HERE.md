# RobotWorld：17 个快反馈任务交接包

给智钦，2026-09-27。

打开 `index.html` 浏览。每个任务单独在 `tasks/Txx/README.md`，完整结构化字段在 `tasks.json`。

## 这包是什么

这是上一轮保留的 **17 个原生任务设置**，不是 17 个已经跑通的 RobotWorld adapter，也不保证全部是高难题。本包新增了资产具体位置、对应资产定义、版本/安装参考、任务入口、原 setting、观测/动作、原评估、接入时不能改丢的条件，以及相关原始源码快照。

模型 USD/URDF/mesh、动作 npz、checkpoint **没有整体装进压缩包**。按 `ASSETS.md` / `assets.json` 从各固定版本上游获取，保留相对引用；`sources/` 是供离线阅读的源码快照，不是完整可安装仓库。环境运行需另行克隆上游仓库。

## 文件导航

- `index.html`：17 题浏览与检索，离线可打开。
- `tasks/T01…T17/README.md`、`spec.json`：逐题接入说明。
- `ASSETS.md` / `assets.json`：资产文件、来源、生成方式和未验证项。
- `ENVIRONMENTS.md`：不同容器/版本的分组和入口。
- `INTEGRATION.md`：接入与验收顺序，原指标与 RobotWorld 指标的边界。
- `tasks.json` / `sources.json`：机器可读的任务与源码映射。
- `sources/`：固定版本的相关源码、配置、资产定义及可取得的许可证。
- `fetch_repositories.py`：可选，只克隆并 checkout 固定源代码，不装依赖、不运行训练、不下载外部 NVIDIA 资产。
- `verify_bundle.py` / `manifest.json`：离线 SHA-256 完整性校验。

## 先怎么做

先接 T01 端托盘、T02 随机乒乓来球、T03 随机接球、T06 漂移、T07 近跌倒恢复，覆盖不同机制。先按原 setting 得到可复现结果，再测 agent。其余任务作为接入优先级稍后的补充，17 题选择保持不变。

不要把所有任务塞进一个新版本容器：此清单包含 Sim 2023.1.0-hotfix.1、4.0、4.1、4.2、4.5 与 5.1 相关项目。优先按作者版本启动，再考虑迁移。

本次只做文件与源码层面的核查。未执行任何 GPU rollout；视频只是部分片段；“保留”不等于“排除了所有简单解”。
