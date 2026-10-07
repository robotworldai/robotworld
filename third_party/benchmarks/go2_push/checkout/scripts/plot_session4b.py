"""Generate all session-4b plots and baseline-vs-pushrecovery comparison plots."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PR_LOG  = "logs/rsl_rl/unitree_go2_flat_pushrecovery/2026-06-02_23-49-30"
BL_LOG  = "logs/rsl_rl/unitree_go2_flat/2026-06-02_11-58-18"
OUT_DIR = "docs/images"

def load(path):
    ea = EventAccumulator(path, size_guidance={'scalars': 0})
    ea.Reload()
    return ea

def sc(ea, tag):
    evs = ea.Scalars(tag)
    return np.array([e.step for e in evs]), np.array([e.value for e in evs])

pr = load(PR_LOG)
bl = load(BL_LOG)
print("TFEvents loaded.")

BLUE   = '#1f77b4'
ORANGE = '#ff7f0e'

def mk(figsize=(12, 4.5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, color='#cccccc', linestyle='--', linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    return fig, ax

def save(name):
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{name}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {OUT_DIR}/{name}")

# 1 — mean reward
fig, ax = mk()
iters, vals = sc(pr, 'Train/mean_reward')
ax.plot(iters, vals, color=BLUE, linewidth=0.8)
best_i = int(np.argmax(vals))
ax.axvline(iters[best_i], color='red', linestyle=':', linewidth=1.2,
           label=f'Best iter {iters[best_i]} (reward {vals[best_i]:.2f})')
ax.legend(fontsize=11)
ax.set_xlabel('Iteration')
ax.set_ylabel('Mean Reward')
ax.set_title('Mean Reward — Push-Recovery (3072 envs, 2500 iters)')
ax.set_xlim(0, iters[-1])
save("pushrecovery_mean_reward.png")

# 2 — mean episode length
fig, ax = mk()
iters, vals = sc(pr, 'Train/mean_episode_length')
ax.plot(iters, vals, color=BLUE, linewidth=0.8)
ax.set_xlabel('Iteration')
ax.set_ylabel('Mean Episode Length (steps)')
ax.set_title('Mean Episode Length — Push-Recovery (3072 envs, 2500 iters)')
ax.set_xlim(0, iters[-1])
save("pushrecovery_mean_episode_length.png")

# 3 — lin vel tracking reward
fig, ax = mk()
iters, vals = sc(pr, 'Episode_Reward/track_lin_vel_xy_exp')
ax.plot(iters, vals, color=BLUE, linewidth=0.8)
ax.set_xlabel('Iteration')
ax.set_ylabel('Reward (lin vel tracking)')
ax.set_title('Linear Velocity Tracking Reward — Push-Recovery')
ax.set_xlim(0, iters[-1])
save("pushrecovery_rew_track_linvel.png")

# 4 — ang vel tracking reward
fig, ax = mk()
iters, vals = sc(pr, 'Episode_Reward/track_ang_vel_z_exp')
ax.plot(iters, vals, color=BLUE, linewidth=0.8)
ax.set_xlabel('Iteration')
ax.set_ylabel('Reward (ang vel tracking)')
ax.set_title('Angular Velocity Tracking Reward — Push-Recovery')
ax.set_xlim(0, iters[-1])
save("pushrecovery_rew_track_angvel.png")

# 5 — curriculum
fig, ax1 = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor('white')
ax1.set_facecolor('white')
ax1.grid(True, color='#cccccc', linestyle='--', linewidth=0.6)
ax1.spines['top'].set_visible(False)

i_imp, v_imp = sc(pr, 'Curriculum/push_impulse_force')
i_sus, v_sus = sc(pr, 'Curriculum/push_sustained_force')
i_ic,  v_ic  = sc(pr, 'Curriculum/push_impulse_count')
i_sc2, v_sc2 = sc(pr, 'Curriculum/push_sustained_count')

l1, = ax1.plot(i_imp, v_imp, color=BLUE,   linewidth=1.4, label='Impulse max force (N)')
l2, = ax1.plot(i_sus, v_sus, color=ORANGE, linewidth=1.4, label='Sustained max force (N)')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Max Push Force (N)')
ax1.set_xlim(0, i_imp[-1])
ax1.set_ylim(0, 135)

ax2 = ax1.twinx()
ax2.spines['top'].set_visible(False)
l3, = ax2.plot(i_ic,  np.cumsum(v_ic),  color=BLUE,   linewidth=0.8, linestyle='--',
               alpha=0.55, label='Impulse triggers (cumulative)')
l4, = ax2.plot(i_sc2, np.cumsum(v_sc2), color=ORANGE, linewidth=0.8, linestyle='--',
               alpha=0.55, label='Sustained triggers (cumulative)')
ax2.set_ylabel('Cumulative Trigger Count', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

lines = [l1, l2, l3, l4]
ax1.legend(lines, [l.get_label() for l in lines], fontsize=10, loc='center right')
ax1.set_title('Push-Recovery Curriculum — Force Ramps and Trigger Counts')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/pushrecovery_curriculum.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  saved {OUT_DIR}/pushrecovery_curriculum.png")

# 6 — baseline vs pushrecovery mean reward
fig, ax = mk()
bi, bv = sc(bl, 'Train/mean_reward')
pi, pv = sc(pr, 'Train/mean_reward')
ax.plot(bi, bv, color=BLUE,   linewidth=0.8, label='Baseline (session 2, 2000 iter)')
ax.plot(pi, pv, color=ORANGE, linewidth=0.8, label='Push-Recovery (session 4b, 2500 iter)')
ax.axvline(bi[-1], color='gray', linestyle=':', linewidth=1.0, label='Baseline end (iter 1999)')
ax.set_xlabel('Iteration')
ax.set_ylabel('Mean Reward')
ax.set_title('Baseline vs Push-Recovery — Mean Reward')
ax.set_xlim(0, max(bi[-1], pi[-1]))
ax.legend(fontsize=11)
save("baseline_vs_pushrecovery_mean_reward.png")

# 7 — baseline vs pushrecovery lin vel tracking
fig, ax = mk()
bi, bv = sc(bl, 'Episode_Reward/track_lin_vel_xy_exp')
pi, pv = sc(pr, 'Episode_Reward/track_lin_vel_xy_exp')
ax.plot(bi, bv, color=BLUE,   linewidth=0.8, label='Baseline (session 2, 2000 iter)')
ax.plot(pi, pv, color=ORANGE, linewidth=0.8, label='Push-Recovery (session 4b, 2500 iter)')
ax.axvline(bi[-1], color='gray', linestyle=':', linewidth=1.0, label='Baseline end (iter 1999)')
ax.set_xlabel('Iteration')
ax.set_ylabel('Reward (lin vel tracking)')
ax.set_title('Baseline vs Push-Recovery — Linear Velocity Tracking Reward')
ax.set_xlim(0, max(bi[-1], pi[-1]))
ax.legend(fontsize=11)
save("baseline_vs_pushrecovery_rew_track_linvel.png")

# 8 — baseline vs pushrecovery episode length
fig, ax = mk()
bi, bv = sc(bl, 'Train/mean_episode_length')
pi, pv = sc(pr, 'Train/mean_episode_length')
ax.plot(bi, bv, color=BLUE,   linewidth=0.8, label='Baseline (session 2, 2000 iter)')
ax.plot(pi, pv, color=ORANGE, linewidth=0.8, label='Push-Recovery (session 4b, 2500 iter)')
ax.axvline(bi[-1], color='gray', linestyle=':', linewidth=1.0, label='Baseline end (iter 1999)')
ax.set_xlabel('Iteration')
ax.set_ylabel('Mean Episode Length (steps)')
ax.set_title('Baseline vs Push-Recovery — Mean Episode Length')
ax.set_xlim(0, max(bi[-1], pi[-1]))
ax.legend(fontsize=11)
save("baseline_vs_pushrecovery_episode_length.png")

print("All 8 plots saved.")
