# `auto_collect_results/` - Automatic Batch Assessment and Results Collection

This catalogue provides a batch of ** scripts for training products (`steps_*_pytorch_model.pt`) **.
Responsible for:

1. Launching of SimplerEnv assessment missions on SLURM clusters in bulk;
2. Scan logs after completion, success rate, output CSV + folding diagram;
3. Maintain ckpt directory (cleaning `.pt`).

Overview of document roles:

| Documentation | Role | Input | Output |
| --- | --- | --- | --- |
| `schedule_widowx_eval.sh` | Batch Schedule ** WidowX / BridgeData v2** (4 Job) Assessment | `<ROOT_BASE>/<DIR_GLOB>/checkpoints/steps_*_pytorch_model.pt` | Each ckpt 4 and `*.log.run1` (written by `star_bridge.sh`) |
| `schedule_google_eval.sh` | Sets a ckpt serial srun up ** Google Robot** for all 8 subassessments | Single `MODEL_PATH` | `client_logs/steps_<step>/*.log` |
| `summarize_widowx_one.sh` | Parsing ** A widowx log under ** Experimental Directory, calling Python out | Experiment Directory Path | `success_summary/raw_success.txt` `success_summary.csv` `success_plot.png` |
| `summarize_widowx_all.sh` | Loop `summarize_widowx_one.sh` on all matching directories | Directory glob under `ROOT_BASE` | `success_summary/` for each directory |
| `plot_widowx_results.py` | Actual drawing: Press step x task for success rate CSV + PNG | `raw_success.txt` | `csv` + `png` |
| `rm_pt.sh` | Clear `.pt` files in glob batch (with dry-run switches) | `ROOT_DIR` + `DIR_GLOB` + `FILE_GLOB` | Delete File |

> The original filename used `windox`, a misspelling of `widowx` (the BridgeData v2 WidowX arm); the rename also corrects this typo.

---

## Typical Workstream

### A. Assessment WidowX (Bridge))

```
[Training outputs steps_*_pytorch_model.pt]
        │
        ▼
schedule_widowx_eval.sh           # Movement control srun，Missing log ckpt I'll run.
        │
        ▼
star_bridge.sh   (Each ckpt 4 individual task × N seed)
        │
        ▼
steps_<step>_pytorch_model_infer_<TASK>-v0.log.run1
        │
        ▼
summarize_widowx_all.sh           # Or a single directory. summarize_widowx_one.sh
        │
        ▼
success_summary/{raw_success.txt, success_summary.csv, success_plot.png}
```

### B. Assessment Google Robot

`schedule_google_eval.sh` targets **one checkpoint** (edit the settings at the top of
`MODEL_DIR` / `step` after direct execution) will parallel srun up 8 to `star_*.sh`
(drawer / move_near / pick_coke_can / put_in_drawer each variant + visual_matching).

---

## How? `0427_oxe_bridge_rt_1_QwenPI_v3`.

`schedule_widowx_eval.sh` has drawn the target directory into parameters. The default value for ** is
`0427_oxe_bridge_rt_1_QwenPI_v3` **, so direct:

```bash
cd examples/SimplerEnv/eval_files/auto_eval_scripts/auto_collect_results

# Modalities 1：Use default DIR_GLOB
bash schedule_widowx_eval.sh

# Modalities 2：Visible Directory Name / glob
bash schedule_widowx_eval.sh '0427_oxe_bridge_rt_1_QwenPI_v3'
bash schedule_widowx_eval.sh '0427_oxe_bridge*'

# Modalities 3：env var Form
DIR_GLOB='0427_oxe_bridge_rt_1_QwenPI_v3' \
SLURM_PARTITION=si SLURM_GRES=gpu:4 \
bash schedule_widowx_eval.sh
```

After running, aggregate results:

```bash
# Single Experiment Directory
bash summarize_widowx_one.sh \
  /mnt/petrelfs/yejinhui/Projects/starVLA/results/Checkpoints/0427_oxe_bridge_rt_1_QwenPI_v3

# Or change. summarize_widowx_all.sh Top DIR_GLOB Backload running
DIR_GLOB='0427_oxe_bridge_rt_1_QwenPI_v3' bash summarize_widowx_all.sh
```

The fusion results will fall.
`<ckpt_dir>/success_summary/{raw_success.txt, success_summary.csv, success_plot.png}`。

---

## Modemable Environmental Variables

`schedule_widowx_eval.sh`：

| Variables | Default value | Annotations |
| --- | --- | --- |
| `ROOT_BASE` | `/mnt/petrelfs/yejinhui/Projects/starVLA/results/Checkpoints` | Experiment Roots Directory |
| `DIR_GLOB` | `0427_oxe_bridge_rt_1_QwenPI_v3` | glob for the experimental subdirectories (also as the first position parameter) |
| `SLURM_PARTITION` | `si` | Partition of srun |
| `SLURM_GRES` | `gpu:4` | srun resources |
| `SCRIPT_PATH` | `.../auto_eval_scripts/star_bridge.sh` | bridge Assessment Entry |

`summarize_widowx_all.sh` / `summarize_widowx_one.sh`：

| Variables | Default value | Annotations |
| --- | --- | --- |
| `ROOT_BASE` | `/mnt/.../results/Checkpoints` | Experiment Roots Directory |
| `DIR_GLOB` | `0427_oxe_bridge_rt_1_QwenPI_v3` | Experiment to scan glob |
| `RM_LOGS` | `false` | Whether success logs were deleted without resolution |
