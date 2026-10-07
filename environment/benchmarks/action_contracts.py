"""Prompt-only action contracts, shared by system instructions and dynamic tools.

Source audit: docs/ACTION_CONTRACT_AUDIT.md. No action or physics implementation here.
"""

COMMON = '''ACTION SEMANTICS AND EXAMPLES
Examples below explain the interface, not a demonstrated solution or collision-free motion.
A control target is not a measured result: actuator limits, contact, gravity and tracking error remain active. Check fresh allowed observations after changing a command. Camera image-left is not necessarily robot/world +Y. Positive rotation follows the right-hand rule about the stated axis; joint signs follow the authored joint axes, including mirrored joints. Do not transfer units, quaternion ordering or zero/hold conventions between benchmarks.
Separate tool calls execute sequentially; use a combined action to request simultaneous channels. Simulator time advances only through the action interface, not while reasoning or computing offline. Feedback code, when enabled, must return the same permitted action type; it cannot directly manipulate the simulator. A successful tool call means execution, not task success.
'''

CONTRACTS = {
'robodojo': '''RoboDojo / move_eef:
- All named x/y/z values are ABSOLUTE WORLD grasp-point coordinates in metres, not deltas, flange positions or camera coordinates. +Z is up; the two robot bases face world +Y; world +X runs laterally. Both measured Cartesian state and commands refer to the point between the jaws; do not add a flange-to-fingertip offset yourself.
- Example: if measured left_z=0.90, targets={"left_z":0.92} requests a 2 cm upward change. Repeating left_z=0.92 requests that same height, not another 2 cm. Unnamed pose coordinates latch their observed values at the start of this call.
- pitch_deg/roll_deg/yaw_deg are ABSOLUTE angle parameters in DEGREES relative to the straight-down reference. The naming is special: pitch rotates about world +X, then roll about world +Y, then yaw about world +Z; Q=Qz(yaw)*Qy(roll)*Qx(pitch)*Q_down. They are not conventional RPY, radians, or angular increments. All three zeros point the outward tool axis down world -Z. pitch=+90,roll=yaw=0 points it toward world +Y. yaw=+10 from the down reference turns the jaw orientation 10 degrees about world +Z, without changing the commanded grasp point.
- Gripper opening: 0 closed, 1 open. Example left_gripper=0 closes, unlike RoboCasa's gripper_close=0. A pose and gripper in ONE call execute arm motion first, jaw change after arrival. Both arms' planned paths are time-aligned; this is different from simultaneous gripper closure in BEHAVIOR/RoboLab/RoboCasa.
- move_eef has no user steps or speed input. The planner and resampling choose a bounded trajectory and count its actual environment steps; a move requiring over 10 seconds is rejected and must be split. Zero coordinate is an absolute coordinate, not hold. Joint angles in observations cannot be commanded through this tool. Check arrival residuals and planner status; bounds alone do not prove reachability or clearance.
''',
'behavior_1k': '''BEHAVIOR-1K / R1Pro:
- left/right_x/y/z are ABSOLUTE EEF positions in the robot ARTICULATION-ROOT frame, metres. quat_xyzw is an absolute root-relative orientation, XYZW; identity=[0,0,0,1]. It is not WXYZ, Euler angles or a rotation increment. +X forward, +Y left, +Z up. Root-relative hand targets move in the world when the base translates/turns; holding them is not holding a fixed world pose.
- Example: measured left position [0.40,0.20,0.80]; targets={"left_z":0.81} requests a 1 cm upward target change. Repeating it holds root-relative z=0.81. It does NOT add 0.81m per tick. A +10-degree root-Z orientation relative to identity is XYZW [0,0,0.08715574,0.9961947]; for a rotated hand compose the desired orientation instead of overwriting with this example.
- base_vx/base_vy are local-body velocity targets in m/s; base_wz is rad/s, positive counterclockwise viewed from above. Limits: +/-0.3m/s and +/-0.5rad/s. Example base_vx=0.1,steps=15 at 30Hz requests forward 0.1m/s for 0.5s; ideal no-slip travel is 5cm, NOT guaranteed displacement. base_wz=0.2 over those 15 steps corresponds ideally to +0.1rad yaw. Zero requests zero speed, not an instantaneous freeze.
- trunk_qpos is the FULL absolute trunk joint vector in observed order, radians, within runtime limits. For one joint changing 0.20 to 0.21 requests +0.01rad (~0.573deg), not 1cm of height. Positive trunk angle does not generically mean upward Cartesian motion.
- left/right_gripper are opening targets: 0 closed, 1 open. They persist across calls. move_robot applies arm, base, trunk and jaws simultaneously; closing begins during movement, not on arrival.
- Ordinary tool calls: steps=1..30 at 30Hz repeats the same assembled action; omitted EEF/trunk coordinates latch MEASURED values at call start, omitted base velocities=0, grippers retain prior commands. Feedback-code callbacks, if enabled: omitted EEF/trunk coordinates instead retain LAST PROGRAM TARGETS (initially measured), omitted base velocities=0 each tick, grippers persist. Neither mode automatically ramps a large absolute target; generate small reference changes when appropriate.
''',
'robocasa': '''RoboCasa / pinned PandaOmron OSC + mobile base:
- Tool fields map to the native 12D input: [eef_dx,dy,dz,rx,ry,rz,gripper_close,base_x,base_y,base_yaw,torso,control_mode]. Motion fields are normalized inputs in [-1,1], NOT absolute poses. EEF observations use base-relative metres and quaternion XYZW; action rotations are ROTATION VECTORS, not quaternions or Euler angles.
- EEF delta translation=0.05*u metres and rotation-vector=0.5*u radians in the ROBOT BASE frame. Delta rotation left-multiplies the chosen base-relative orientation: R_next=Exp([0.5*u_rot]x)*R_reference. Example eef_delta=[0,0,0.1,0,0,0] requests a +5mm base-Z goal offset on EACH step. eef_delta=[0,0,0,0,0,0.1] requests +0.05rad (~2.865deg) about base +Z per step. This is not world-up if the reference base is tilted.
- control_mode=0 computes each arm goal from the currently ACHIEVED pose; 1 adds to the last DESIRED base-relative goal. Repeating z=0.1 for 10 steps requests repeated +5mm offsets; with mode1 the desired target accumulates +5cm (before limits), with mode0 actual progress depends on tracking and is not guaranteed 5cm. Zero arm delta with mode0 reanchors to achieved pose; with mode1 retains the desired root-relative goal as the base moves.
- Base inputs are velocity-controller fractions, not metre displacements. The pinned native controller rotates the XY input using theta=current_base_yaw-initial_base_yaw: v_forward=u_x*cos(theta)+u_y*sin(theta), v_side=-u_x*sin(theta)+u_y*cos(theta). The initial-frame slide actuator limits scale these to 1m/s per unit; yaw input scales to 1.5rad/s per unit. At theta=0, base_motion=[0.1,0,0] requests +0.1m/s along initial base +X; [0,0,0.1] requests +0.15rad/s about +Z. At theta=pi/2, [0.1,0,0] instead commands initial slide velocities [0,-0.1]m/s. This is the pinned implementation's sign convention; do not replace it with an assumed world/local transform. These are servo requests, not achieved travel or braking guarantees.
- Torso is an upward SLIDE joint, metres, NOT a revolute angle. torso=u requests q_goal=q_measured+0.05*u metres each tick; nominal travel range 0..0.34m. Example torso=0.1 requests +5mm from current measured height per tick; it is not a 10cm absolute height or 0.1m/s command. Zero reanchors to measured height.
- gripper_close=1 closes, 0 opens; binary target persists. All supplied channels act together, including closure during approach. Unspecified EEF/base/torso inputs=0. Default control_mode is 1 for move_base/move_torso, 0 for move_eef/set_gripper/move_robot unless explicitly supplied. The standard Gym profile is 20Hz: steps=10 holds the native vector for 0.5s, recomputing deltas every tick. No interpolation or collision avoidance is implied.
''',
'ai_cps': '''AI-CPS / Franka incremental joint controller:
- arm_action has exactly seven simultaneous entries in panda_joint1..7 order. Each entry u in [-1,1] is a normalized JOINT-TARGET INCREMENT: q_target_next=clip(q_target_previous+0.125*u, joint_limits), before accounting for upstream action noise. Rotational units after conversion are radians; this is not absolute qpos, EEF delta, torque or velocity. Fingers are task-controlled and cannot be commanded here.
- Example only joint1=+0.04, others=0: nominal joint1 target rises +0.005rad (~0.286deg) per tick; held for 10 ticks it accumulates +0.05rad (~2.865deg), unless clamped/noisy. Negative u reduces the corresponding joint coordinate. Mirrored/rotated kinematic axes mean this does not guarantee Cartesian +X or upward movement.
- Zero means no nominal addition to the PREVIOUS TARGET, not hold measured qpos or stop momentum. To seek an absolute angle, compute a bounded error relative to the previous target and account for fresh measurement, noise and limits; do not send the desired angle as u.
- ~60Hz native control (obs.dt is authoritative): 10 steps is about 1/6 simulated second. move_joints repeats the same increment, while a feedback program, if enabled, chooses a fresh increment at every tick. Both consume the same action-step budget. cancel_action in task34 acknowledges contact; after cancellation each noisy target increment is limited to +/-0.00625rad/tick, not a guaranteed actual joint velocity.
''',
}


def action_contract(benchmark, mode=None):
    if benchmark == 'robolab':
        modes = {
            'joint_position': 'ACTIVE MODE joint_position: joint_positions are seven ABSOLUTE panda_joint1..7 angles, radians. Example target joint1=0.21 instead of measured 0.20 requests +0.01rad (~0.573deg); repeat 0.21 to retain the same target, not another increment. Supply the full seven-vector, preserving other measured/desired coordinates. Zero vector requests seven zero angles, not hold.',
            'absolute_ik': 'ACTIVE MODE absolute_ik: position is an ABSOLUTE gripper base_link flange position in robot-root metres, quaternion_wxyz is absolute orientation in WXYZ. Example measured ee_pos=[0.40,0.10,0.50], position=[0.40,0.10,0.51] requests +1cm in root +Z. Repeating the same position holds that target. Identity WXYZ=[1,0,0,0], not [0,0,0,1]; identity is not necessarily a downward grasp. ee_pos/ee_quat are the matching IK frame; eef_pos/eef_quat are a DIFFERENT reporting frame, not the commanded flange.',
            'relative_ik': 'ACTIVE MODE relative_ik: delta_pose=[dx,dy,dz,rx,ry,rz] is root-axis translation plus rotation vector, scaled by 0.5 per step; translations become metres and rotations radians. Example [0,0,0.02,0,0,0] requests +1cm root-Z offset from the currently measured flange each tick. Repeating ten ticks recomputes that offset ten times; actual travel is not guaranteed 10cm. [0,0,0,0,0,0.02] requests +0.01rad about root +Z each tick (delta rotation left-multiplies current orientation). Zero delta asks IK to retain current measured pose; it is not a zero absolute pose.',
        }
        if mode not in modes: raise ValueError('An explicit RoboLab controller mode is required')
        detail = '''RoboLab / fixed-base DROID:\n'''+modes[mode]+'''
All Cartesian targets are robot articulation-root relative, NOT camera/world positions; root +Z is up in the configured fixed mounting. The controlled point is the gripper base_link, not the fingertips. Position limits/reach and contact are not solved by the IK interface.
Ordinary steps=1..30 at 15Hz; 15 steps=1 simulated second. The same assembled vector is repeated. Omitted absolute coordinates/joints latch measured values at call start; omitted relative delta=0. gripper_close=0 opens and 1 closes; its command persists. move_robot applies arm and gripper together, so closure begins while moving. No mobile base/torso tool or task-solving planner exists.
'''
    else:
        detail = CONTRACTS[benchmark]
    return COMMON + detail


def describe_tools(tools, contract):
    """Return independent specs; attach the identical contract to action tools."""
    return [{**tool, 'description': tool['description'] + ('\n\n' + contract if tool['name'] not in ('observe', 'cancel_action') else '')}
            for tool in tools]
