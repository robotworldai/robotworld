# World 文件放置指南

| 目录 | 放什么 | 不放什么 |
|---|---|---|
| codex/ | 本地原版 Codex 源码 | 我们的工具、密钥、编译产物 |
| third_party/benchmarks/ | 各项目 checkout 原版源码；其外层 README、版本配置、已有 Docker/compat 包装 | 资产本体、缓存、episode 输出 |
| third_party/dependencies/ | IsaacLab、CuRobo、XPolicyLab 上游源码 | Python 安装环境、镜像 tar |
| environment/ | 自己维护的 adapters、tools、prompt、评测桥、核验器；核心环境已有 Docker 配置和下载器 | 上游源码副本、资产本体 |
| Assets/<benchmark>/ | 场景、机器人、物体、纹理、官方数据与权重 | 源码、运行视频、登录凭证 |
| reports/ | 核验结论、资产迁移记录、直接 MP4 索引 | 当作任务成功率的启动测试 |
| var/datasets/ | 迁移期间兼容旧入口；最终链接到 Assets | 新建另一份资产副本 |
| var/build/ | 编译结果与镜像构建材料 | 登录凭证 |
| var/cache/ | 可重建缓存 | 唯一一份重要结果 |
| var/runs/ | 视频、轨迹、events、官方评分 | 凭证 |
| var/auth/ | 本机登录信息 | GitHub/HF 发布内容 |
| upload-github/ | 之前准备的发布快照 | 当作最新开发目录使用 |
| upload-hf/ | 可再分发的既有 RoboDojo 资产发布材料 | BEHAVIOR 资产或任何密钥 |

每个我们维护的模块目录均有 README。上游 checkout、缓存内层和每次生成的运行目录不注入 README，避免修改上游或历史证据；遵循其父目录指引。

本地已有 Git 仓库用 local clone，工作树独立、不可变 Git 对象共享硬链接，避免再下载历史。不是给源码文件做硬链接，不会通过编辑一份工作树改动另一份。没有复制原目录中未跟踪的巨量资产、安装环境或输出；原目录也没有删除。

Docker 镜像由 Docker 管理，不在 World 再保存一份镜像 tar；World 中保留 Dockerfile、构建/启动脚本和 digest。既有 RoboDojo 本机 snapshot 构建脚本仍依赖外部安装环境，不能把本次源码归档称为完全自包含的安装包。BEHAVIOR 默认启动路径已切到 World 内的 checkout。

新增 benchmark：建立 third_party 包装目录并锁版本；在 environment 下分别建 benchmarks、containers、scenarios、datasets 的专属实现；资产进入 Assets，运行结果进入 var。不要在上游源码中增加自定义 policy。

本轮题集在 `environment/validation/selected-tasks.json`；正常模型评测选题在 `environment/evaluation/suites.json`，四个核心 benchmark 的默认题集已按新清单更新为 15/10/10/10。旧案例只保留显式选择入口。

上游自带资产在 Assets 的收集副本使用硬链接，保持 checkout 的原路径、原字节和 Git 状态；不增加第二份物理数据。它们按不可变文件处理。对外导出代码时按 `environment/datasets/embedded-assets.json` 排除相应上游资产文件，不直接上传整个目录。


现有 Docker 文件有两种位置：早期 RoboDojo/BEHAVIOR 配置在 `environment/containers/<bench>/`，其他项目主要在 `third_party/benchmarks/<bench>/docker/`。统一用户入口是 `scripts/eval/<bench>.sh build`；不要按目录名猜启动命令。`third_party` 包装目录中的 `docker/compat/robotworld` 属于 World 自有代码，其内 `checkout/` 才是保持不改动的上游仓库。保留这些已验证的加载路径；本轮把资产本体集中到 Assets，不为了目录外观重新改动所有导入路径。
