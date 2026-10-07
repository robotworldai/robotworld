# 上游不可修改与 GitHub 发布

## Codex

源码固定使用 /opt/robotworld/World/codex。由它构建的二进制通过绝对路径配置，产物放 var/build/codex/<commit>/。runner 校验来源与版本，缺失即报错，不回退到 PATH 或全局安装版本。

使用原 Cargo.lock 的 locked 构建，构建缓存/target 输出指向 var；构建前后检查 git status。不得修改任何 Codex 源码、Cargo.toml、Cargo.lock 或配置 schema。配置和密钥从我们维护的外部配置/运行时注入。

本次只读盘点 HEAD: 8ae55c863db26d417e83390c5854f1144114276b。此值是盘点记录，不替代后续正式锁定清单。

## Benchmark

2026-09-25 用户明确授权例外：“可以在外部写兼容代码”。针对 Isaac Sim 6.0.1，允许 `environment/benchmarks/robodojo/compat/` 中显式启用的运行时接口适配，包括 monkey patch；仍不得修改上游源码文件。必须记录启用状态并验证任务语义，不能默认宣称与旧引擎评测等价。此例外不放宽 Codex 源码不可修改的约束，优先于下文原有运行时补丁禁令。

third_party/benchmarks/<id>/README.md 保存来源说明，checkout/ 放本机干净克隆；既有 RoboDojo/ 目录保持原路径。依赖源码放 third_party/dependencies/<id>/checkout/。版本见 third_party/sources.local.json，未来可明确登记 submodule。不能往上游目录增加 adapter、policy 包或替换文件。

原方案“往 XPolicyLab.policy 下添加入口包”修订为外部安装包/官方插件发现接口。RoboDojo 固定模块导入是否允许外部 namespace 路径需验证；若不允许，采用外部 runner 调用其公开生命周期和评测 API。不得通过模块遮蔽、修改 __path__ 或 monkey patch 伪装已满足无修改要求。若公开 API 无法复现官方评测生命周期，先报告阻塞，不宣称官方等价。

现有 /opt/robotworld/RoboDojo 可作只读参考；盘点 HEAD 为 ee67a1468510da7624a089164402359f2afc72c8，工作区已有改动。正式评测另取固定提交的干净 clone，不能重置用户现有工作区。

## 发布方式

优先提交自己的代码、Markdown、配置模板与来源锁定清单；各上游用 URL + commit 引用，也可在确认来源与许可后登记 Git submodule。不得把第三方代码整体复制提交。codex/ 可在日后初始化 World 仓库时登记为固定版本 submodule；当前不移动它、不初始化父 Git、不创建含猜测 URL 的 .gitmodules。

来源清单必须记录 repository URL、commit、子模块版本、许可证、获取方式、本地覆盖路径。RoboLab 名称对应的具体仓库尚未确认；其余 URL 和兼容版本也在接入前核实，不填写猜测链接。

上游代码只读挂载，日志/资产/编译缓存通过官方配置写入独立可写目录。若上游强制写源码目录，需要先确认官方支持的替代路径，不能承诺所有环境已兼容。
