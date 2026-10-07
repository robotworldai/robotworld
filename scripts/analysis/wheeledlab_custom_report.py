"""Read-only HTML review of the four custom driving calibration episodes."""
import argparse
import base64
import html
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    ('courtyard-gpt6-01', '园区配送', '基础目标', '绕箱、0.15m 缓坡、定向停车',
     '按序检查点；终点距离≤0.65m、朝向误差≤25°、速度≤0.12m/s 持续0.8s。'),
    ('hairpins-gpt6-01', '高架连续回头弯', '困难目标', '1.35m 窄路、升高1m、两个180°弯',
     '按序通过带高度要求的检查点；终点距离≤0.40m、朝向≤12°、低速保持0.8s。'),
    ('gate-gpt6-01', '动态闸门与停靠', '困难目标', '真实运动闸门、停车等待、限速、连续弯道',
     '停车线内低速停留0.5s；净空>0.5m时过闸、窄口≤0.65m/s；按序检查点；终点≤0.32m、朝向≤10°、低速保持1s。'),
    ('drift-gpt6-01', '受扰变摩擦漂移', '困难目标', '后驱、四段摩擦、运行中速度/偏航扰动',
     '同一圈通过所有检查点并过终点线；两个弯各连续≥0.35s侧滑，速度≥0.7m/s、前向速度≥0.5m/s、侧滑角0.25–0.70rad。'),
]


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def track_svg(spec, states):
    path = spec['path']
    lo = [min(p[i] for p in path) - 2 for i in range(2)]
    hi = [max(p[i] for p in path) + 2 for i in range(2)]
    width, height = hi[0] - lo[0], hi[1] - lo[1]
    points = lambda rows: ' '.join(f'{p[0]:.4f},{-p[1]:.4f}' for p in rows)
    out = [f'<svg role="img" aria-label="灰色道路与蓝色真实轨迹，俯视图" viewBox="{lo[0]} {-hi[1]} {width} {height}">',
           '<rect x="-50" y="-50" width="100" height="100" fill="#f0f4f2"/>',
           f'<polyline points="{points(path)}" fill="none" stroke="#b9c5c1" stroke-width="{spec["width"]}" stroke-linejoin="round"/>']
    for box in spec['obstacles']:
        x, y, _ = box['center']; w, h, _ = box['size']
        out.append(f'<rect x="{x-w/2}" y="{-y-h/2}" width="{w}" height="{h}" fill="#a28566"/>')
    if states:
        out.append(f'<polyline points="{points([s["position"] for s in states])}" fill="none" stroke="#087fc4" stroke-width="0.065"/>')
        x, y, _ = states[-1]['position']
        out.append(f'<circle cx="{x}" cy="{-y}" r="0.16" fill="#e06228"/>')
    for cp in spec['checkpoints']:
        x, y, _ = cp['position']
        out.append(f'<circle cx="{x}" cy="{-y}" r="0.09" fill="#fff" stroke="#576c67" stroke-width=".025"/>')
    out.append('</svg>')
    return ''.join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, default=ROOT / 'var/runs/docker/wheeledlab/custom-v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'robotworld-wheeledlab-custom-scenes.html')
    args = parser.parse_args()
    cards = []
    for dirname, title, tier, subtitle, criterion in CASES:
        run = args.runs / dirname
        result = read(run / 'result.json', {})
        interrupted = read(run / 'interrupted.json', {})
        spec = read(run / 'scenario.json', {})
        score = result.get('robotworld', interrupted.get('last_score', {}))
        states = []
        trace = run / 'events/scoring.jsonl'
        if trace.exists():
            for line in trace.read_text().splitlines():
                try:
                    states.append(json.loads(line)['payload']['state'])
                except (ValueError, KeyError):
                    continue  # Live report may see a partially written last line.
        label = '成功' if result.get('success') is True else '失败' if result.get('success') is False else '已中止（不计输赢）' if interrupted else '未完成'
        step = result.get('control_steps', states[-1]['step'] if states else 0)
        figure = ''
        image = run / 'video/terrain-overview.png'
        if image.exists():
            encoded = base64.b64encode(image.read_bytes()).decode()
            figure = f'<img alt="{title}实际Isaac渲染" src="data:image/png;base64,{encoded}">'
        links = []
        for name, target in [('视频', run / 'video/camera.mp4'), ('完整结果', run / 'result.json'),
                             ('逐步评分轨迹', trace), ('去图像工具日志', run / 'events/no-images/tools.jsonl')]:
            if interrupted and name == '视频': continue  # Old interrupted MP4 lacks its final container index.
            if target.exists():
                href = html.escape(os.path.relpath(target, args.output.parent), quote=True)
                links.append(f'<a href="{href}">{name}</a>')
        details = f'检查点 {score.get("checkpoints_passed", "—")}/{score.get("checkpoints_total", "—")}；停车保持 {score.get("parking_hold_s", 0):.2f}s'
        if 'drift' in spec:
            details = f'完成圈数 {score.get("completed_laps", 0)}；本圈两弯最长连续侧滑 {score.get("drift_continuous_best_s", [])}s'
        failure = score.get('failure_reason')
        cards.append(f'''<article><div class="eyebrow">{tier} · {html.escape(spec.get('case', dirname))}</div>
<h2>{title}</h2><p class="subtitle">{subtitle}</p><div class="figures">{figure}{track_svg(spec, states) if spec else ''}</div>
<div class="result">{label} · {step}/2000 步 · {step*.02:.2f}s 仿真</div>
<p><b>成功条件：</b>{criterion}且全过程无安全失败。</p><p>{html.escape(details)}</p>
{f'<p class="failure">终止原因：{html.escape(failure)}</p>' if failure else ''}
<nav>{' '.join(links)}</nav></article>''')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RobotWorld · 四个驾驶场景</title><style>
*{box-sizing:border-box}body{margin:0;background:#ecf1ef;color:#19352f;font:16px/1.7 system-ui,sans-serif}main{max-width:1440px;margin:auto;padding:40px 28px}h1{font-size:36px;line-height:1.25;margin:12px 0}h2{font-size:25px;margin:7px 0}.eyebrow{font-size:13px;color:#44766a;letter-spacing:.04em}.intro{max-width:1050px}.notice{background:#dce9e3;border-left:4px solid #37876c;padding:16px 20px;margin:24px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}article{background:white;border:1px solid #d6e0db;border-radius:12px;padding:22px;min-width:0}.subtitle{margin-top:0;color:#596f66}.figures{display:grid;grid-template-columns:1fr 1fr;gap:8px}.figures img,.figures svg{width:100%;aspect-ratio:4/3;object-fit:contain;background:#f0f4f2;border-radius:6px}.result{font-size:19px;font-weight:700;margin-top:18px}.failure{color:#a64029}nav{display:flex;gap:16px;flex-wrap:wrap}a{color:#126aa1}footer{margin-top:32px;font-size:14px;color:#597168}@media(max-width:850px){.grid{grid-template-columns:1fr}main{padding:20px 16px}h1{font-size:28px}}
</style><main><div class="eyebrow">ROBOTWORLD / WHEELEDLAB / CUSTOM V1</div><h1>四个驾驶场景 · 地形、动作与完成证据</h1>
<p class="intro">旧状态观测条件的首次校准记录。本地源码 Codex + GPT-6（gpt-6-astra），每题上限2000控制步，50Hz，最多40秒仿真。真实碰撞道路、坡面、障碍和运动闸门；灰色是道路、蓝色是实际轨迹、橙色是最终位置。渲染图为本轮场地全景，视频为连续控制步录像。</p>
<p>当前默认接口已改为<a href="third_party/benchmarks/wheeledlab/robotworld/ONBOARD.md">车载相机＋编码器＋IMU</a>，移除了地图、全局位姿、检查点和闸门提示。下方旧成绩不代表新版成绩。漂移轮在切换时中止，旧MP4未完成封装，只保留可读轨迹。</p>
<div class="notice"><b>实验边界：</b>这是 RobotWorld 自定义任务，不是官方 WheeledLab 分数。策略获得模拟器状态与公共地图，不是纯视觉。1道基础、3道困难是设计目标，不预设模型输赢。采用 Isaac6.0.1 / IsaacLab2.2 实验兼容环境；上游车辆源码未改，不宣称与官方4.5物理等价。</div>
<section class="grid">''' + ''.join(cards) + '''</section><footer>共同失败条件：安全包络越出道路、侵入障碍/闸门、跌落、翻覆、轨迹不连续或超时。几何安全判据不等于接触力测量。所有成功必须由冻结评分器根据真实状态计算，不能由模型宣称完成。一次seed=7运行不能代表总体成功率。LLM思考时暂停仿真；code回调每个控制步反馈，模型不是50Hz墙钟推理。视频及日志链接依赖同目录World运行产物。协议详见 third_party/benchmarks/wheeledlab/robotworld/README.md。</footer></main></html>'''
    args.output.write_text(document)
    print(args.output)


if __name__ == '__main__':
    main()
