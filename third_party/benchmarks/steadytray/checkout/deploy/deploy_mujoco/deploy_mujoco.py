import time
import argparse
import os
import sys
import mujoco.viewer
import mujoco
import numpy as np
import torch
from collections import deque

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Add parent directory to path for common imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.policy_runner import compute_policy_action, detect_policy_type, detect_encoder_obs_size
from scripts.config import Config
from scipy.spatial.transform import Rotation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_object_pose(data, model, object_half_height=0.05):
    """
    Get object pose in camera frame from MuJoCo simulation.
    
    Args:
        data: MuJoCo data object
        model: MuJoCo model object
        object_half_height: Half of the object height in meters (default: 0.05m = 5cm for half height)
                           This is added to get the top surface position instead of center
    
    Returns:
        object_obs: Position + quaternion observation array (7,)
    """
    # Get camera pose in world frame
    cam_site_id = model.site("d435_camera_frame").id
    cam_pos_world = data.site_xpos[cam_site_id]
    cam_rot_world = data.site_xmat[cam_site_id].reshape(3, 3)

    # Get object pose in world frame (center of object)
    object_body_id = model.body("object").id
    object_pos_world = data.xpos[object_body_id]
    object_rot_world = data.xmat[object_body_id].reshape(3, 3)
    
    # Add half height to get top surface position
    # The offset is in the object's local frame (z-axis points up in object frame)
    top_surface_offset_local = np.array([0, 0, object_half_height], dtype=np.float32)
    top_surface_offset_world = object_rot_world @ top_surface_offset_local
    object_top_pos_world = object_pos_world + top_surface_offset_world

    # Build transformation matrices (using top surface position)
    object_world_transform = np.eye(4)
    object_world_transform[:3, :3] = object_rot_world
    object_world_transform[:3, 3] = object_top_pos_world  # Use top surface position

    camera_world_transform = np.eye(4)
    camera_world_transform[:3, :3] = cam_rot_world
    camera_world_transform[:3, 3] = cam_pos_world

    # Transform object pose to camera frame
    object_camera_transform = np.linalg.inv(camera_world_transform) @ object_world_transform
    object_camera_pos = object_camera_transform[:3, 3]  # This is now the top surface position
    object_camera_rotation = Rotation.from_matrix(object_camera_transform[:3, :3])
    
    # Convert to wxyz quaternion format
    object_camera_quat_xyzw = object_camera_rotation.as_quat()
    object_camera_quat = np.array([object_camera_quat_xyzw[3], object_camera_quat_xyzw[0], 
                                   object_camera_quat_xyzw[1], object_camera_quat_xyzw[2]], dtype=np.float32)

    # Position + quaternion (3+4)
    object_obs = np.concatenate([object_camera_pos, object_camera_quat], axis=0).astype(np.float32)
    
    return object_obs



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='g1 deploy mujoco')
    parser.add_argument('--policy', type=str, default="exported/policy_9999.pt",
                       help='Direct path to policy file')
    parser.add_argument('--config', type=str, default="deploy/configs/g1_29dof_walk.yaml",
                       help='Direct path to config file (overrides default config)')
    parser.add_argument('--encoder_seq_len', type=int, default=32,
                       help='Encoder sequence length (number of history frames for distillation policies)')

    args = parser.parse_args()

    # Load configuration using shared Config class
    config = Config(args.config)
    
    # Get policy path
    if args.policy is not None:
        policy_path = args.policy
    else:
        raise ValueError("Policy path must be provided via command line argument --policy")

    if os.path.exists(policy_path):
        print(f"Using policy: {policy_path}")
    else:
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    # define context variables
    action = np.zeros(config.num_actions, dtype=np.float32)
    obs = np.zeros(config.num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(config.xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = config.simulation_dt

    default_angles = config.default_angles[config.policy_to_robot]
    target_dof_pos = default_angles.copy()

    frame_stack = deque(maxlen=5)
    for _ in range(5):
        frame_stack.append(obs.copy())
        mujoco.mj_step(m, d) 


    # Load policy
    policy = torch.jit.load(policy_path)
    policy_type = detect_policy_type(policy)
    print(f"Policy type: {policy_type}")

    # Auto-detect encoder observation size for distillation policies
    encoder_obs_dim = None
    
    if policy_type == 'distillation':
        encoder_obs_dim = detect_encoder_obs_size(policy)

    print(f"Using MuJoCo ground-truth object observations")

    # Initialize encoder frame stack for distillation policies
    encoder_frame_stack = deque(maxlen=args.encoder_seq_len)
    if encoder_obs_dim is not None:
        for _ in range(args.encoder_seq_len):
            encoder_frame_stack.append(np.zeros(encoder_obs_dim, dtype=np.float32))

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Set up camera to follow the robot
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        # Track the pelvis/base body (usually body id 1, adjust if needed)
        # You can find the correct body by looking at the robot's URDF/XML structure
        viewer.cam.trackbodyid = 1  # Changed from 0 (world) to 1 (pelvis/base)
        viewer.cam.distance = 2.5   # Distance from robot (increased for better view)
        viewer.cam.elevation = -20  # Camera angle (negative looks down)
        viewer.cam.azimuth = 90     # Side view angle
        viewer.cam.lookat[:] = [0, 0, 0.5]  # Look at point offset
        
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < config.simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:7 + config.num_actions], config.kps, np.zeros_like(config.kds), d.qvel[6:6 + config.num_actions], config.kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % config.control_decimation == 0:
                # Apply control signal here.

                start_compute = time.time()
                # Get sensor data (explicitly use float32 for consistency with real robot)
                qj = d.qpos[7:7 + config.num_actions].astype(np.float32)
                dqj = d.qvel[6:6 + config.num_actions].astype(np.float32)
                quat = d.qpos[3:7].astype(np.float32)
                omega = d.qvel[3:6].astype(np.float32)

                # Get object observations from MuJoCo simulation
                object_obs = None
                
                if encoder_obs_dim is not None:
                    object_obs = get_object_pose(d, m)

                # Compute policy action using shared function
                action, target_dof_pos = compute_policy_action(
                    policy=policy,
                    frame_stack=frame_stack,
                    qj=qj,
                    dqj=dqj,
                    quat=quat,
                    omega=omega,
                    cmd=config.cmd_init,
                    previous_action=action,
                    config=config,
                    object_obs=object_obs,
                    policy_type=policy_type,
                    encoder_frame_stack=encoder_frame_stack
                )
                compute_time = time.time() - start_compute
                if compute_time > config.control_dt:
                    print(f"Warning: Policy compute time {compute_time:.6f} seconds exceeds control_dt {config.control_dt} seconds")

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
