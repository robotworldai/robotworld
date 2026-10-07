"""do-mpc based nonlinear MPC core for benchmark baselines."""

from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

try:  # do-mpc is optional unless NMPC is used.
    import casadi as ca
    import do_mpc

    _DO_MPC_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local do-mpc environment.
    ca = None
    do_mpc = None
    _DO_MPC_IMPORT_ERROR = exc


@dataclass
class NMPCObjectiveCfg:
    """Objective weights for target-position NMPC."""

    terminal_pb: float = 1.0
    terminal_vb: float = 1.0
    terminal_theta: float = 1.0
    terminal_vrz: float = 1.0
    stage_pb: float = 1.0
    stage_vb: float = 0.25
    stage_theta: float = 0.0
    stage_vrz: float = 0.5
    rterm_u: float = 0.05


@dataclass
class NMPCConstraintsCfg:
    """State and input bounds for target-position NMPC."""

    ball_position_min: float = 0.0
    ball_position_max: float = 0.7
    theta_min: float = -40.0 * np.pi / 180.0
    theta_max: float = 40.0 * np.pi / 180.0
    acceleration_min: float = -0.5
    acceleration_max: float = 0.5
    velocity_min: float | None = None
    velocity_max: float | None = None


@dataclass
class NMPCSolverCfg:
    """do-mpc solver configuration."""

    n_horizon: int = 30
    t_step: float = 1.0 / 60.0
    n_robust: int = 0
    store_full_solution: bool = True
    ipopt_print_level: int = 0
    print_time: bool = False
    suppress_ipopt_output: bool = True


@dataclass
class NMPCControllerCfg:
    """Complete configuration for a single do-mpc controller."""

    gravity: float = 9.81
    plank_length: float = 1.06
    rope_length: float = 0.9
    ball_mass: float = 0.0005
    ball_radius: float = 0.023
    ball_inertia_ratio: float = 0.4
    initial_goal: float = 0.35
    objective: NMPCObjectiveCfg = field(default_factory=NMPCObjectiveCfg)
    constraints: NMPCConstraintsCfg = field(default_factory=NMPCConstraintsCfg)
    solver: NMPCSolverCfg = field(default_factory=NMPCSolverCfg)


class TargetPositionNMPC:
    """Single-env nonlinear MPC using benchmark velocity-interface dynamics."""

    def __init__(self, cfg: NMPCControllerCfg):
        require_do_mpc()
        self.cfg = cfg
        self.position_goal = float(cfg.initial_goal)
        self.start_mpc = True
        self._setup_controller()

    def reset(self):
        """Reset warm-start history on the next solve."""
        self.start_mpc = True
        try:
            self.mpc.reset_history()
        except Exception:
            pass

    def set_goal(self, goal: float):
        """Set the target ball position used by the TVP objective."""
        self.position_goal = float(goal)

    def make_step(self, state: np.ndarray, goal: float | None = None) -> tuple[float, float, float, str]:
        """Solve one NMPC step.

        Args:
            state: ``[pb, vb, theta, vrz]``.
            goal: optional target position for this solve.

        Returns:
            acceleration command, compute time, success flag, solver status.
        """
        if goal is not None:
            self.set_goal(float(goal))

        x0 = np.asarray(state, dtype=np.float64).reshape(4, 1)
        if self.start_mpc:
            self.mpc.reset_history()
            self.mpc.x0 = x0
            self.mpc.set_initial_guess()
            self.start_mpc = False

        start_time = time.perf_counter()
        try:
            u0 = self.mpc.make_step(x0)
            compute_time = time.perf_counter() - start_time
            status = self._solver_status()
            success = 1.0 if status == "unknown" or _status_success(status) else 0.0
            return float(np.asarray(u0).reshape(-1)[0]), float(compute_time), success, status
        except Exception as exc:
            compute_time = time.perf_counter() - start_time
            self.start_mpc = True
            return 0.0, float(compute_time), 0.0, f"exception:{type(exc).__name__}"

    def _setup_controller(self):
        self.model = do_mpc.model.Model(model_type="continuous")
        self._set_model()
        self._set_controller()

    def _set_model(self):
        cfg = self.cfg
        pb = self.model.set_variable(var_type="_x", var_name="pb", shape=(1, 1))
        vb = self.model.set_variable(var_type="_x", var_name="vb", shape=(1, 1))
        theta = self.model.set_variable(var_type="_x", var_name="theta", shape=(1, 1))
        vrz = self.model.set_variable(var_type="_x", var_name="vrz", shape=(1, 1))
        u = self.model.set_variable(var_type="_u", var_name="u", shape=(1, 1))
        beta = self.model.set_variable(var_type="_z", var_name="beta", shape=(1, 1))
        omega = self.model.set_variable(var_type="_z", var_name="omega", shape=(1, 1))
        self.pg = self.model.set_variable(var_type="_tvp", var_name="pg", shape=(1, 1))

        effective_mass = cfg.ball_mass * (1.0 + cfg.ball_inertia_ratio)
        ball_acc = (cfg.ball_mass * (pb - cfg.plank_length) * omega**2 - cfg.ball_mass * cfg.gravity * ca.sin(theta))
        ball_acc = ball_acc / effective_mass

        self.model.set_rhs(var_name="pb", expr=vb)
        self.model.set_rhs(var_name="vb", expr=ball_acc)
        self.model.set_rhs(var_name="theta", expr=omega)
        self.model.set_rhs(var_name="vrz", expr=u)

        self.model.set_alg(
            expr_name="geometry_constraints",
            expr=ca.vertcat(
                ca.sin(beta) - (cfg.plank_length / cfg.rope_length) * (1.0 - ca.cos(theta)),
                cfg.plank_length * omega * ca.cos(beta - theta) + vrz * ca.cos(beta),
            ),
        )
        self.model.setup()

    def _set_controller(self):
        cfg = self.cfg
        obj = cfg.objective
        con = cfg.constraints
        solver = cfg.solver

        self.mpc = do_mpc.controller.MPC(self.model)
        self.mpc.set_param(
            n_horizon=int(solver.n_horizon),
            t_step=float(solver.t_step),
            n_robust=int(solver.n_robust),
            store_full_solution=bool(solver.store_full_solution),
        )
        if solver.suppress_ipopt_output:
            try:
                self.mpc.settings.supress_ipopt_output()
            except Exception:
                pass

        error = self.model.x["pb"] - self.pg
        mterm = (
            obj.terminal_pb * error**2
            + obj.terminal_vb * self.model.x["vb"] ** 2
            + obj.terminal_theta * self.model.x["theta"] ** 2
            + obj.terminal_vrz * self.model.x["vrz"] ** 2
        )
        lterm = (
            obj.stage_pb * error**2
            + obj.stage_vb * self.model.x["vb"] ** 2
            + obj.stage_theta * self.model.x["theta"] ** 2
            + obj.stage_vrz * self.model.x["vrz"] ** 2
        )
        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        self.mpc.set_rterm(u=float(obj.rterm_u))

        self.mpc.bounds["lower", "_x", "pb"] = float(con.ball_position_min)
        self.mpc.bounds["upper", "_x", "pb"] = float(con.ball_position_max)
        self.mpc.bounds["lower", "_x", "theta"] = float(con.theta_min)
        self.mpc.bounds["upper", "_x", "theta"] = float(con.theta_max)
        self.mpc.bounds["lower", "_u", "u"] = float(con.acceleration_min)
        self.mpc.bounds["upper", "_u", "u"] = float(con.acceleration_max)
        if con.velocity_min is not None:
            self.mpc.bounds["lower", "_x", "vrz"] = float(con.velocity_min)
        if con.velocity_max is not None:
            self.mpc.bounds["upper", "_x", "vrz"] = float(con.velocity_max)

        tvp_template = self.mpc.get_tvp_template()

        def tvp_fun(_t_now):
            for horizon_idx in range(int(solver.n_horizon) + 1):
                tvp_template["_tvp", horizon_idx, "pg"] = self.position_goal
            return tvp_template

        self.mpc.set_tvp_fun(tvp_fun)
        self.mpc.setup()

    def _solver_status(self) -> str:
        stats = getattr(self.mpc, "solver_stats", None)
        if isinstance(stats, dict):
            return str(stats.get("return_status", stats.get("success", "unknown")))
        return "unknown"


class NMPCWorker(mp.Process):
    """Worker process owning one do-mpc controller."""

    def __init__(self, remote, cfg_dict: dict):
        super().__init__()
        self.remote = remote
        self.cfg_dict = cfg_dict

    def run(self):
        try:
            try:
                controller = TargetPositionNMPC(_controller_cfg_from_dict(self.cfg_dict))
                init_error = None
            except Exception as exc:
                controller = None
                init_error = f"init_exception:{type(exc).__name__}"
            while True:
                command, data = self.remote.recv()
                if controller is None:
                    if command == "exit":
                        self.remote.send(True)
                        break
                    if command == "step":
                        self.remote.send((0.0, 0.0, 0.0, init_error))
                    else:
                        self.remote.send(False)
                    continue
                if command == "reset":
                    controller.reset()
                    self.remote.send(True)
                elif command == "set_goal":
                    controller.set_goal(float(data))
                    self.remote.send(True)
                elif command == "step":
                    state, goal = data
                    self.remote.send(controller.make_step(state, goal=goal))
                elif command == "exit":
                    self.remote.send(True)
                    break
                else:
                    self.remote.send((0.0, 0.0, 0.0, f"unknown_command:{command}"))
        except (EOFError, BrokenPipeError):
            pass
        finally:
            try:
                self.remote.close()
            except Exception:
                pass


class NMPCControllerPool:
    """Manage one NMPC controller per parallel environment."""

    def __init__(self, cfg: NMPCControllerCfg, num_envs: int, use_multiprocessing: bool = True):
        require_do_mpc()
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.use_multiprocessing = bool(use_multiprocessing and self.num_envs > 1)
        self._closed = False
        self.controllers: list[TargetPositionNMPC] | None = None
        self.remotes = None
        self.workers: list[NMPCWorker] = []

        cfg_dict = asdict(cfg)
        if self.use_multiprocessing:
            self.remotes, worker_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
            self.workers = [NMPCWorker(remote, cfg_dict) for remote in worker_remotes]
            for worker in self.workers:
                worker.daemon = True
                worker.start()
            for remote in worker_remotes:
                remote.close()
        else:
            self.controllers = [TargetPositionNMPC(cfg) for _ in range(self.num_envs)]

    def reset(self, env_ids: Sequence[int] | None = None):
        env_ids = self._env_ids(env_ids)
        if self.use_multiprocessing:
            for env_id in env_ids:
                self.remotes[env_id].send(("reset", None))
            for env_id in env_ids:
                self.remotes[env_id].recv()
        else:
            for env_id in env_ids:
                self.controllers[env_id].reset()

    def set_goal(self, goals: np.ndarray, env_ids: Sequence[int] | None = None):
        env_ids = self._env_ids(env_ids)
        goals = np.asarray(goals, dtype=np.float64).reshape(self.num_envs)
        if self.use_multiprocessing:
            for env_id in env_ids:
                self.remotes[env_id].send(("set_goal", goals[env_id]))
            for env_id in env_ids:
                self.remotes[env_id].recv()
        else:
            for env_id in env_ids:
                self.controllers[env_id].set_goal(float(goals[env_id]))

    def make_step(self, states: np.ndarray, goals: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        states = np.asarray(states, dtype=np.float64).reshape(self.num_envs, 4)
        if goals is None:
            goal_values = [None for _ in range(self.num_envs)]
        else:
            goal_values = list(np.asarray(goals, dtype=np.float64).reshape(self.num_envs))

        if self.use_multiprocessing:
            for env_id in range(self.num_envs):
                self.remotes[env_id].send(("step", (states[env_id], goal_values[env_id])))
            results = [self.remotes[env_id].recv() for env_id in range(self.num_envs)]
        else:
            results = [
                self.controllers[env_id].make_step(states[env_id], goal=goal_values[env_id])
                for env_id in range(self.num_envs)
            ]
        u, compute_time, success, status = zip(*results)
        return (
            np.asarray(u, dtype=np.float32).reshape(self.num_envs, 1),
            np.asarray(compute_time, dtype=np.float32).reshape(self.num_envs),
            np.asarray(success, dtype=np.float32).reshape(self.num_envs),
            list(status),
        )

    def close(self, timeout_s: float = 2.0):
        if self._closed:
            return
        if self.use_multiprocessing and self.remotes is not None:
            for remote in self.remotes:
                try:
                    remote.send(("exit", None))
                except Exception:
                    pass
            for remote in self.remotes:
                try:
                    if remote.poll(timeout_s):
                        remote.recv()
                except Exception:
                    pass
            for worker in self.workers:
                worker.join(timeout=timeout_s)
            for worker in self.workers:
                if worker.is_alive():
                    worker.terminate()
            for worker in self.workers:
                worker.join(timeout=timeout_s)
            for worker in self.workers:
                if worker.is_alive() and hasattr(worker, "kill"):
                    worker.kill()
            for worker in self.workers:
                worker.join(timeout=timeout_s)
            for remote in self.remotes:
                try:
                    remote.close()
                except Exception:
                    pass
        self._closed = True
        self.remotes = None
        self.workers = []

    def __del__(self):
        try:
            self.close(timeout_s=0.2)
        except Exception:
            pass

    def _env_ids(self, env_ids: Sequence[int] | None) -> list[int]:
        if env_ids is None:
            return list(range(self.num_envs))
        return [int(env_id) for env_id in env_ids]


def require_do_mpc():
    """Raise a clear error if do-mpc/casadi are unavailable."""
    if _DO_MPC_IMPORT_ERROR is not None:
        raise RuntimeError(
            "NMPC baselines require optional dependencies 'do_mpc' and 'casadi'. "
            "Run NMPC scripts in an environment where both packages are installed."
        ) from _DO_MPC_IMPORT_ERROR


def _status_success(status: str) -> bool:
    status_lower = str(status).lower()
    return "success" in status_lower or "succeeded" in status_lower or status_lower == "true"


def _controller_cfg_from_dict(data: dict) -> NMPCControllerCfg:
    return NMPCControllerCfg(
        gravity=float(data["gravity"]),
        plank_length=float(data["plank_length"]),
        rope_length=float(data["rope_length"]),
        ball_mass=float(data["ball_mass"]),
        ball_radius=float(data["ball_radius"]),
        ball_inertia_ratio=float(data["ball_inertia_ratio"]),
        initial_goal=float(data["initial_goal"]),
        objective=NMPCObjectiveCfg(**data["objective"]),
        constraints=NMPCConstraintsCfg(**data["constraints"]),
        solver=NMPCSolverCfg(**data["solver"]),
    )
