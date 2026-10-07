'use strict';
const $ = id => document.getElementById(id);
const node = (tag, text, cls) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; };
const number = (v, digits = 2) => Number.isFinite(v) ? v.toFixed(digits) : '--';
const endings = {give_up: '模型主动结束', timeout: '达到 600 秒预算', action_budget: '动作预算耗尽', tool_budget: '工具预算耗尽', environment_end: '环境结束', agent_completed: '模型结束回合', infrastructure_error: '基础设施错误'};
let data, task, activeEvent = null, camera = 'head_640x480';
const video = $('video');

function actionLabel(event) {
  if (event.kind === 'observe') return '观察环境';
  const a = event.arguments;
  if ((a.base || []).some(v => v !== 0)) return (a.base[0] || a.base[1]) ? '底盘平移' : '底盘转向';
  return `${a.arm === 'left' ? '左臂' : '右臂'} EEF 控制`;
}
function badge(text, cls) { return node('span', text, `badge ${cls || ''}`); }
function metric(label, value, suffix) {
  const box = node('div', undefined, 'metric'), val = node('div', value, 'metric-value');
  if (suffix) val.append(node('small', suffix));
  box.append(node('div', label, 'metric-label'), val); return box;
}
function observationOf(event) { return task.observations.find(o => String(o.observation_id) === event.observation_id); }
function selectCamera(name, reset = false) {
  camera = name;
  document.querySelectorAll('[data-camera]').forEach(b => { b.classList.toggle('selected', b.dataset.camera === name); b.disabled = !task.videos[b.dataset.camera]; });
  const source = task.videos[name], at = reset ? 0 : video.currentTime;
  video.pause();
  video.hidden = !source;
  $('video-empty').hidden = !!source;
  $('video-stamp').textContent = name === 'head_640x480' ? '30 FPS · 640 × 480' : '30 FPS · 128 × 128';
  if (!source) { video.removeAttribute('src'); video.load(); return; }
  video.src = source;
  video.onloadedmetadata = () => { video.currentTime = Math.min(at, Math.max(0, video.duration - 1 / 30)); video.playbackRate = Number($('speed').value); };
  $('playback-label').textContent = `${number(at)} / ${number(task.sim_seconds)} s`;
}
function selectTask(id) {
  task = data.tasks.find(t => t.id === id) || data.tasks[0];
  activeEvent = null;
  document.querySelectorAll('.task-tab').forEach(b => b.setAttribute('aria-selected', String(Number(b.dataset.id) === task.id)));
  $('task-activity').textContent = `TASK ${String(task.id).padStart(2, '0')} / ${task.activity || 'PENDING'}`;
  $('task-title').textContent = task.title;
  $('download').href = task.download;
  $('metrics').replaceChildren(metric('原始 Q 分数', task.score_valid ? number(task.score, 2) : '--', task.score_valid ? '已验证记录' : '未定分'),
    metric('动作请求', String(task.action_calls), '/ 40'), metric('物理仿真时间', number(task.sim_seconds), '秒'),
    metric('EEF 未到位', String(task.unreached), `/ ${task.action_calls} 次动作`));
  const outcome = $('outcome'); outcome.replaceChildren();
  const pending = task.status !== 'completed';
  outcome.append(badge(task.score_valid ? '评分记录有效' : '尚无有效评分', task.score_valid ? 'green' : 'amber'));
  outcome.append(node('h4', pending ? (task.status === 'environment_ready' ? '模型回合进行中' : '环境准备中') : endings[task.termination] || '回合结束', 'outcome-title'));
  outcome.append(node('p', task.score_valid ? (task.success ? '原始 checker 判定任务成功。' : '原始 checker 未判定成功。本结果只反映当前观测、控制与预算配置。') : '环境或结果尚未完成验收，不把未完成记录计为模型 0 分。', 'outcome-note'));
  if (task.stop_reason) {
    outcome.append(node('p', 'MODEL REPORTED REASON', 'stop-label'), node('p', task.stop_reason.reason, 'stop-text'));
    if (task.stop_reason.hindsight) { const details = node('details', undefined, 'goal'); details.append(node('summary', '模型回顾原文'), node('p', task.stop_reason.hindsight, 'stop-text')); outcome.append(details); }
  } else if (task.termination === 'timeout') outcome.append(node('p', '控制器在预算耗尽时中断模型；不是模型调用 give_up。', 'stop-text'));
  if (task.api_span_seconds != null) outcome.append(node('p', `API 交互跨度约 ${number(task.api_span_seconds, 1)} 秒。视频只有 ${number(task.sim_seconds, 1)} 秒物理运动。`, 'fine'));
  const labels = {environment_valid: '原生环境 / checker', request_boundary_valid: '当前图片 / 工具边界', video_valid: '连续录像 / 帧数校验'};
  $('checks').replaceChildren(...Object.entries(labels).map(([key, label]) => { const row = node('div', undefined, 'check'), value = task.checks[key]; row.append(node('span', label), node('strong', value === true ? '通过' : value === false ? '未通过' : '待定', value !== true ? 'pending' : '')); return row; }));
  $('goal').replaceChildren(...(task.instructions.length ? task.instructions : ['任务观测尚未就绪']).map(x => node('p', x)));
  $('prompt').textContent = Object.entries(task.prompt).map(([key, text]) => `${key}\n\n${text}`).join('\n\n');
  $('search').value = ''; $('filter').value = 'all';
  $('timeline').replaceChildren(...task.events.map(event => { const b = node('button'); b.className = event.kind === 'observe' ? 'observe' : event.feedback.reached === false ? 'unreached' : ''; b.title = `${event.number}. ${actionLabel(event)} · step ${event.start_step}–${event.end_step}`; b.setAttribute('aria-label', b.title); b.dataset.event = event.number; b.onclick = () => selectEvent(event.number, true); return b; }));
  $('event-count').textContent = `${task.events.length} 条记录`;
  selectCamera('head_640x480', true);
  renderOutline();
  if (task.events.length) selectEvent(task.events[0].number, false);
  else $('event-detail').replaceChildren(node('p', '模型尚未产生工具调用，稍后更新报告。', 'empty'));
}
function renderOutline() {
  const q = $('search').value.trim().toLowerCase(), filter = $('filter').value;
  const filtered = task.events.filter(e => (!q || JSON.stringify(e).toLowerCase().includes(q) || actionLabel(e).includes(q)) &&
    (filter === 'all' || filter === e.kind || (filter === 'unreached' && e.feedback.reached === false)));
  $('match-count').textContent = `${filtered.length} / ${task.events.length} 条`;
  $('outline').replaceChildren(...filtered.map(event => {
    const b = node('button'), head = node('span', undefined, 'outline-top'); b.dataset.event = event.number;
    head.append(node('span', String(event.number).padStart(3, '0')), node('span', event.kind.toUpperCase()));
    b.append(head, node('span', actionLabel(event), 'outline-name'), node('span', event.feedback.reached === false ? `EEF 未到位 · ${number(event.feedback.position_error_m * 1000, 1)} mm` : `step ${event.start_step} → ${event.end_step}`, 'outline-info'));
    b.onclick = () => selectEvent(event.number, true); b.setAttribute('aria-current', String(event.number === activeEvent)); return b;
  }));
  if (!filtered.length) { $('outline').append(node('p', '没有匹配记录', 'empty')); $('event-detail').replaceChildren(node('p', '当前筛选没有匹配记录。', 'empty')); activeEvent = null; }
  else if (!filtered.some(e => e.number === activeEvent)) selectEvent(filtered[0].number, false);
}
function selectEvent(id, seek) {
  const event = task.events.find(e => e.number === id); if (!event) return;
  activeEvent = id;
  document.querySelectorAll('[data-event]').forEach(b => b.setAttribute('aria-current', String(Number(b.dataset.event) === id)));
  if (seek && video.getAttribute('src')) { video.pause(); if (video.readyState) video.currentTime = event.video_time; else video.addEventListener('loadedmetadata', () => { video.currentTime = event.video_time; }, {once: true}); }
  const body = $('event-detail'), head = node('div', undefined, 'event-head'), title = node('div');
  title.append(node('p', `EVENT ${String(id).padStart(3, '0')} / ${event.kind.toUpperCase()}`, 'eyebrow'), node('h3', actionLabel(event)), node('p', `${event.source}${event.native_line ? ` · actions.jsonl 第 ${event.native_line} 行` : ''}`, 'event-meta'));
  head.append(title, badge(event.error ? '请求被拒绝' : event.kind === 'observe' ? '不推进物理' : event.feedback.reached === false ? 'EEF 未到位' : 'EEF 到位', event.feedback.reached === false || event.error ? 'amber' : 'green'));
  const counters = node('div', undefined, 'event-counters'); counters.append(node('span', `执行 ${event.executed_steps} 控制步`), node('span', `仿真 ${number(event.start_step / 30)} → ${number(event.end_step / 30)} s`));
  if (event.checker) counters.append(node('span', `动作后 Q = ${number(event.checker.q_score)} · 仅供评审`));
  const grid = node('div', undefined, 'event-grid');
  for (const [label, value] of [['工具参数', event.arguments], ['执行反馈', event.error || (Object.keys(event.feedback).length ? event.feedback : '纯观察，不提交动作')]]) { const pane = node('section'); pane.append(node('h4', label), node('pre', JSON.stringify(value, null, 2), 'code')); grid.append(pane); }
  body.replaceChildren(head, counters, grid);
  const obs = observationOf(event);
  if (obs) {
    body.append(node('h4', `动作后的原始视觉观察 #${obs.observation_id} · 每路 128 × 128`, 'observation-title'));
    const gallery = node('div', undefined, 'gallery');
    for (const [key, label] of [['head', '头部 RGB'], ['left_wrist', '左腕 RGB'], ['right_wrist', '右腕 RGB']]) {
      if (!obs.images[key]) continue;
      const a = node('a'); a.href = obs.images[key]; a.target = '_blank'; a.rel = 'noopener'; const img = node('img'); img.src = obs.images[key]; img.alt = `观察 ${obs.observation_id} ${label}`; img.loading = 'lazy'; a.append(img, node('span', label)); gallery.append(a);
    }
    body.append(gallery);
  }
  const raw = node('details', undefined, 'event-raw'); raw.append(node('summary', '展开完整展示记录 / 实际控制向量'), node('pre', JSON.stringify(event, null, 2), 'code')); body.append(raw);
  body.scrollTop = 0;
}
video.addEventListener('timeupdate', () => {
  if (!task) return;
  $('playback-label').textContent = `${number(video.currentTime)} / ${number(task.sim_seconds)} s`;
  if (!video.paused) { const frame = Math.floor(video.currentTime * 30) + 1; const event = task.events.find(e => e.executed_steps > 0 && frame > e.start_step && frame <= e.end_step); if (event && event.number !== activeEvent) {
    if (!$('outline').querySelector(`[data-event="${event.number}"]`)) { $('search').value = ''; $('filter').value = 'all'; renderOutline(); }
    selectEvent(event.number, false);
  } }
});
$('speed').onchange = () => { video.playbackRate = Number($('speed').value); };
document.querySelectorAll('[data-camera]').forEach(b => { b.onclick = () => selectCamera(b.dataset.camera); });
$('search').oninput = renderOutline; $('filter').onchange = renderOutline;
addEventListener('hashchange', () => { if (data) selectTask(Number(location.hash.replace('#task-', ''))); });
const embeddedReport = document.getElementById('report-data');
const reportPromise = embeddedReport ? Promise.resolve(JSON.parse(embeddedReport.textContent)) : fetch('data.json', {cache: 'no-store'}).then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); });
reportPromise.then(value => {
  data = value; $('headline').replaceChildren(document.createTextNode(String(data.tasks.filter(t => t.score_valid).length).padStart(2, '0')), node('small', '/ 04 已完成'));
  $('updated').textContent = `数据快照 ${new Date(data.generated_at).toLocaleString('zh-CN', {timeZone: 'Asia/Shanghai', hour12: false})} CST`;
  $('task-tabs').replaceChildren(...data.tasks.map(t => { const b = node('button', undefined, 'task-tab'); b.dataset.id = t.id; b.setAttribute('role', 'tab'); const top = node('span', undefined, 'tab-top'); top.append(node('span', `TASK 0${t.id}`), badge(t.score_valid ? `Q ${number(t.score)}` : '进行中', t.score_valid && t.success ? 'green' : 'amber')); b.append(top, node('span', t.title, 'tab-title')); b.onclick = () => { if (location.hash === `#task-${t.id}`) selectTask(t.id); else location.hash = `task-${t.id}`; }; return b; }));
  $('task-tabs').setAttribute('role', 'tablist'); $('loading').hidden = true; $('workspace').hidden = false; selectTask(Number(location.hash.replace('#task-', '')) || 1);
}).catch(error => { $('loading').textContent = `报告加载失败：${error.message}。请刷新重试。`; });
