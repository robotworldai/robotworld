# Copyright (c) 2024, Mickyas Tamiru Asfaw. MIT License.
"""Sim-to-sim validation: run the Isaac-Lab-trained policies in MuJoCo.

The policy was trained in Isaac Sim (PhysX). Running the exact same exported actor
in MuJoCo, an independent physics engine, tests whether the behaviour survives a
change of simulator, which is a strong proxy for sim-to-real transfer. The actor is
a small ELU MLP with no observation normalization, so it runs here in pure numpy
(weights in policies/*.npz, exported by export_policies.py).

Everything is matched to the Isaac Lab task:
  - observation: base angular velocity (base frame), projected gravity (base frame),
    front-thigh joint positions, joint velocities [FL_thigh, FR_thigh, rl_wheel,
    rr_wheel], last action, and (velocity tasks) the 3-value command.
  - action -> targets: thigh position = 0.5 * a; wheel velocity = scale * a
    (scale 5 for balance, 12 for velocity).
  - control at 50 Hz (decimation 4 over a 200 Hz / dt=0.005 sim); 20 s = 1000 steps.

Usage (with the MuJoCo env):
  python sim2sim.py --task balance --episodes 20
  python sim2sim.py --task velocity --cmd_x 0.5 --viewer
  python sim2sim.py --task velocity_rough --episodes 20
"""

import argparse
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# task -> (mjcf, policy npz, wheel action scale, has velocity command)
TASKS = {
    "balance": ("wheeled_quadruped.xml", "balance.npz", 5.0, False),
    "velocity": ("wheeled_quadruped.xml", "velocity.npz", 12.0, True),
    "balance_rough": ("wheeled_quadruped_rough.xml", "balance_rough.npz", 5.0, False),
    "velocity_rough": ("wheeled_quadruped_rough.xml", "velocity_rough.npz", 12.0, True),
}

THIGH_JOINTS = ["robot1_front_left_thigh_joint", "robot1_front_right_thigh_joint"]
WHEEL_JOINTS = ["robot1_rl_wheel_joint", "robot1_rr_wheel_joint"]
ALL_JOINTS = THIGH_JOINTS + WHEEL_JOINTS  # Isaac joint order
BASE_BODY = "robot1_base_footprint"
DECIMATION = 4
MAX_STEPS = 1000            # 20 s at 50 Hz control
FALL_HEIGHT = 0.4
TILT_COS = 0.5             # cos(60 deg): fall when projected-gravity z rises above -0.5


def elu(x):
    return np.where(x > 0.0, x, np.exp(np.minimum(x, 0.0)) - 1.0)


def load_policy(npz):
    d = np.load(os.path.join(HERE, "policies", npz))
    n = int(d["n_layers"])
    layers = [(d[f"W{i}"], d[f"b{i}"]) for i in range(n)]

    def forward(obs):
        x = obs.astype(np.float32)
        for i, (W, b) in enumerate(layers):
            x = W @ x + b
            if i < n - 1:
                x = elu(x)
        return x  # deterministic action = policy mean

    return forward, layers[0][0].shape[1]


class Robot:
    def __init__(self, mjcf):
        self.model = mujoco.MjModel.from_xml_path(os.path.join(HERE, mjcf))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
        self.root_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "root")]
        self.thigh_qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in THIGH_JOINTS]
        self.all_vadr = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ALL_JOINTS]
        self.act_id = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j.replace("robot1_", "")) for j in ALL_JOINTS]
        self.key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        self._make_hfield(seed=0)

    def _make_hfield(self, seed):
        """Fill the 'rough' height field with gentle, wheel-traversable bumps.

        MuJoCo initializes hfield data to zero (flat). We write a smoothed random
        field so the terrain matches the Isaac rough task: low-amplitude roughness
        (elevation up to the geom's 0.06 m), no stairs. Deterministic per seed.
        """
        m = self.model
        hid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_HFIELD, "rough")
        if hid < 0:
            return
        nr, nc = int(m.hfield_nrow[hid]), int(m.hfield_ncol[hid])
        rng = np.random.default_rng(seed)

        def smooth(a, k):
            for _ in range(k):
                a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5.0
            return a

        def norm(a):
            a = a - a.min()
            return a / (a.max() + 1e-9)

        # rolling hills (low frequency, large amplitude) + small bumps (high frequency),
        # so the terrain reads as clearly rough yet stays wheel-traversable.
        coarse = rng.random((12, 12))
        hills = np.kron(coarse, np.ones((nr // 12 + 1, nc // 12 + 1)))[:nr, :nc]
        hills = smooth(hills, 25)
        bumps = smooth(rng.random((nr, nc)), 5)
        h = norm(0.7 * norm(hills) + 0.3 * norm(bumps))
        adr = int(m.hfield_adr[hid])
        m.hfield_data[adr:adr + nr * nc] = h.astype(np.float32).ravel()

    def reset(self, rng):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        d, qa = self.data, self.root_qadr
        # small reset randomization (matches Isaac reset events)
        d.qpos[qa:qa + 2] += rng.uniform(-0.05, 0.05, 2)          # x, y
        yaw = rng.uniform(-0.1, 0.1)
        d.qpos[qa + 3:qa + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        for a in self.thigh_qadr:
            d.qpos[a] += rng.uniform(-0.05, 0.05)
        mujoco.mj_forward(self.model, self.data)

    def observe(self, last_action, command):
        m, d = self.model, self.data
        R = d.xmat[self.base_id].reshape(3, 3)
        proj_grav = R.T @ np.array([0.0, 0.0, -1.0])
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.base_id, vel6, 1)  # local frame
        ang_b = vel6[0:3]
        thigh_pos = np.array([d.qpos[a] for a in self.thigh_qadr])   # rel to default (0)
        joint_vel = np.array([d.qvel[a] for a in self.all_vadr])
        obs = np.concatenate([ang_b, proj_grav, thigh_pos, joint_vel, last_action])
        if command is not None:
            obs = np.concatenate([obs, command])
        return obs.astype(np.float32)

    def apply(self, action, wheel_scale):
        d = self.data
        d.ctrl[self.act_id[0]] = 0.5 * action[0]           # FL thigh position target
        d.ctrl[self.act_id[1]] = 0.5 * action[1]           # FR thigh position target
        d.ctrl[self.act_id[2]] = wheel_scale * action[2]   # rl wheel velocity target
        d.ctrl[self.act_id[3]] = wheel_scale * action[3]   # rr wheel velocity target

    def fallen(self):
        base_z = self.data.qpos[self.root_qadr + 2]
        proj_grav_z = (self.data.xmat[self.base_id].reshape(3, 3).T @ np.array([0, 0, -1.0]))[2]
        return base_z < FALL_HEIGHT or proj_grav_z > -TILT_COS

    def base_lin_vel_b(self):
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, vel6, 1)
        return vel6[3:6]

    def base_ang_vel_b(self):
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, vel6, 1)
        return vel6[0:3]


def run_episode(robot, forward, wheel_scale, command, rng, record_track=False):
    robot.reset(rng)
    last_action = np.zeros(4, np.float32)
    steps = 0
    lin_err = ang_err = nsamp = 0.0
    for t in range(MAX_STEPS):
        obs = robot.observe(last_action, command)
        action = forward(obs)
        robot.apply(action, wheel_scale)
        last_action = action
        for _ in range(DECIMATION):
            mujoco.mj_step(robot.model, robot.data)
        steps += 1
        if record_track and command is not None and not robot.fallen():
            lin_err += abs(command[0] - robot.base_lin_vel_b()[0])
            ang_err += abs(command[2] - robot.base_ang_vel_b()[2])
            nsamp += 1
        if robot.fallen():
            break
    track = None
    if record_track and nsamp > 0:
        track = (lin_err / nsamp, ang_err / nsamp)
    return steps, track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="balance")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--viewer", action="store_true", help="Interactive MuJoCo viewer (loops).")
    ap.add_argument("--cmd_x", type=float, default=0.5, help="Commanded forward velocity (velocity tasks).")
    ap.add_argument("--cmd_yaw", type=float, default=0.0, help="Commanded yaw rate (velocity tasks).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mjcf, npz, wheel_scale, has_cmd = TASKS[args.task]
    forward, obs_dim = load_policy(npz)
    robot = Robot(mjcf)
    command = np.array([args.cmd_x, 0.0, args.cmd_yaw], np.float32) if has_cmd else None
    exp_dim = 16 + (3 if has_cmd else 0)
    assert obs_dim == exp_dim, f"policy obs_dim {obs_dim} != expected {exp_dim}"
    rng = np.random.default_rng(args.seed)

    print(f"task={args.task}  mjcf={mjcf}  policy={npz}  obs_dim={obs_dim}  wheel_scale={wheel_scale}"
          + (f"  command=({args.cmd_x}, 0, {args.cmd_yaw})" if has_cmd else ""))

    if args.viewer:
        import mujoco.viewer
        robot.reset(rng)
        last_action = np.zeros(4, np.float32)
        with mujoco.viewer.launch_passive(robot.model, robot.data) as v:
            step = 0
            while v.is_running():
                obs = robot.observe(last_action, command)
                action = forward(obs)
                robot.apply(action, wheel_scale)
                last_action = action
                for _ in range(DECIMATION):
                    mujoco.mj_step(robot.model, robot.data)
                v.sync()
                step += 1
                if robot.fallen() or step >= MAX_STEPS:
                    robot.reset(rng)
                    last_action = np.zeros(4, np.float32)
                    step = 0
        return

    lengths, lin_errs, ang_errs = [], [], []
    for e in range(args.episodes):
        steps, track = run_episode(robot, forward, wheel_scale, command, rng, record_track=has_cmd)
        lengths.append(steps)
        if track is not None:
            lin_errs.append(track[0]); ang_errs.append(track[1])
    lengths = np.array(lengths)
    print(f"\n=== MuJoCo sim-to-sim: {args.task} ({args.episodes} episodes) ===")
    print(f"  mean episode length : {lengths.mean():.1f} / {MAX_STEPS}   (min {lengths.min()}, max {lengths.max()})")
    print(f"  survived full episode: {(lengths >= MAX_STEPS).mean() * 100:.0f}% of episodes")
    if lin_errs:
        print(f"  lin-vel tracking MAE : {np.mean(lin_errs):.3f} m/s")
        print(f"  yaw-rate tracking MAE: {np.mean(ang_errs):.3f} rad/s")


if __name__ == "__main__":
    main()
