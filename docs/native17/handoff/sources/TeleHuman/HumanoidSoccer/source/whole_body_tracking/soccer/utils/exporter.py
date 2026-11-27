# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch

import onnx

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

from soccer.tasks.tracking.mdp import MotionCommand


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
    motion_name=None
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose, motion_name)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False, motion_name=None):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion")
        # import ipdb; ipdb.set_trace()
        if len(cmd.motion.joint_pos.shape) == 2:  # Single motion.
            self.joint_pos = cmd.motion.joint_pos.to("cpu")
            self.joint_vel = cmd.motion.joint_vel.to("cpu")
            self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
            self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
            self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
            self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
            self.time_step_total = self.joint_pos.shape[0]
        elif len(cmd.motion.joint_pos.shape) == 3:  # Multi-motion.
            # Strip extension to match motion_name entries (stored without suffix).
            motion_name_no_ext = motion_name.split(".")[0] if motion_name else motion_name
            idx = cmd.motion.motion_name.index(motion_name_no_ext)
            self.joint_pos = cmd.motion.joint_pos[idx][:cmd.motion.motion_lengths[idx]].to("cpu")
            self.joint_vel = cmd.motion.joint_vel[idx][:cmd.motion.motion_lengths[idx]].to("cpu")
            self.body_pos_w = cmd.motion.body_pos_w[idx][:cmd.motion.motion_lengths[idx]].to("cpu")
            self.body_quat_w = cmd.motion.body_quat_w[idx][:cmd.motion.motion_lengths[idx]].to("cpu")
            self.body_lin_vel_w = cmd.motion.body_lin_vel_w[idx][:cmd.motion.motion_lengths[idx]].to("cpu")
            self.body_ang_vel_w = cmd.motion.body_ang_vel_w[idx][:cmd.motion.motion_lengths[idx]].to("cpu")
            self.time_step_total = self.joint_pos.shape[0]           


    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(self.normalizer(x)),
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
            self.time_step_total * torch.ones_like(time_step_clamped, dtype=torch.float32).unsqueeze(-1),
        )

    def forward_lstm(self, x_in, h_in, c_in, time_step):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(x),
            h,
            c,
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
            self.time_step_total * torch.ones_like(time_step_clamped, dtype=torch.float32).unsqueeze(-1),
        )

    def export(self, path, filename):
        self.to("cpu")
        self.eval()

        if self.is_recurrent and getattr(self, "rnn_type", "").lower() == "lstm":
            class _LstmExportWrapper(torch.nn.Module):
                def __init__(self, parent):
                    super().__init__()
                    self.parent = parent

                def forward(self, obs, h_in, c_in, time_step):
                    return self.parent.forward_lstm(obs, h_in, c_in, time_step)

            wrapper = _LstmExportWrapper(self)

            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            time_step = torch.zeros(1, 1)
            torch.onnx.export(
                wrapper,
                (obs, h_in, c_in, time_step),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in", "time_step"],
                output_names=[
                    "actions",
                    "h_out",
                    "c_out",
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                    "time_step_total",
                ],
                dynamic_axes={},
            )
        else:
            obs = torch.zeros(1, self.actor[0].in_features)
            time_step = torch.zeros(1, 1)
            torch.onnx.export(
                self,
                (obs, time_step),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "time_step"],
                output_names=[
                    "actions",
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                ],
                dynamic_axes={},
            )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)
    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": env.observation_manager.active_terms["policy"],
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
