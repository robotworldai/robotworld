# 资产发布与一键下载

代码、Docker 配方、资产清单放 GitHub；场景大文件放 Hugging Face **dataset** 仓库。当前仅包含 RoboDojo `arx_x5/1/match_and_pick_from_conveyor_0` 的 216 个资产，约 1.1 GB；不下载完整 benchmark。

## 用户：下载资产

在项目根目录执行（将 OWNER/REPO 替换为发布后的真实 HF 仓库）：

```bash
bash environment/datasets/download_assets.sh --repo-id OWNER/REPO
```

脚本自动创建项目内的专用 Python venv 并安装 huggingface_hub，下载后逐个校验大小和 SHA256。依赖 Python 3.11+、venv/pip。已校验通过的文件直接跳过；中断后重跑相同命令，下载库复用本地缓存。公开仓库无需额外配置凭证；私有仓库使用自己的 HF 登录或 HF_TOKEN，勿写进仓库。

文件自动落到现有 Docker runner 所需的位置：

```text
var/datasets/robodojo/match_and_pick_from_conveyor_0/Assets/
```

无需额外运行 `backup/restore-local-layout.py`。下载成功只表示资产包与清单一致，不等于 Docker/驱动/模型认证已配置完毕。已知源资产仍有 4 条外部/未解析材质引用，详见 reference-audit.json；本包不是已验证的完全离线渲染闭包。

只验证已有资产，不联网、不安装依赖：

```bash
python3 environment/datasets/download_assets.py --verify-only
# 校验待上传 HF 的独立本地副本：
python3 environment/datasets/download_assets.py --verify-only \
  --destination upload-hf/robodojo/match_and_pick_from_conveyor_0/Assets
```

发布后推荐固定 HF 完整 commit：

```bash
bash environment/datasets/download_assets.sh --repo-id OWNER/REPO --revision FULL_COMMIT_SHA
```

也可以将实际仓库名和 commit 写进 `manifests/robodojo-conveyor.json` 的 `repo_id`、`revision`，之后用户只需：

```bash
bash environment/datasets/download_assets.sh
```

当前 repo_id/revision 刻意留空，尚未声称 HF 资产已发布。未指定 revision 时使用 main，但依然必须通过 GitHub 中固定的内容哈希校验。

## 维护者：上传 Hugging Face

本机已准备独立发布目录 `upload-hf`。将此目录的**内容**上传到 HF dataset 仓库根目录，路径必须保持：

```text
README.md
robodojo/match_and_pick_from_conveyor_0/
├── Assets/...
├── asset-manifest.json
└── reference-audit.json
```

创建实际的 HF dataset 仓库后，安装 huggingface_hub 并登录自己的账号，再执行：

```bash
hf auth login
hf upload OWNER/REPO upload-hf . --repo-type dataset
```

这里只给出发布命令，没有执行创建仓库或上传。资产按原文件分发，不重压缩、不修改 USD、纹理或物理参数；来源/许可保持原有约束。未来增加其他 benchmark 时，在 HF 增加各自目录、在 GitHub 增加独立清单，避免每次下载所有资产。

下载实现采用 HF 官方 `hf_hub_download`、dataset 类型、revision 和 local_dir，参见 [下载文档](https://huggingface.co/docs/huggingface_hub/guides/download)；发布命令参见 [上传文档](https://huggingface.co/docs/huggingface_hub/guides/upload)。
