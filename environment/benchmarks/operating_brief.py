"""Public scene/embodiment notes, with the same information contract across suites.

No live USD traversal, object truth, randomized parameters or evaluator output is
used to build these notes. Exact action maps remain owned by each adapter.
"""

SCENES = {
    'robodojo': (
        'A tabletop manipulation workspace with two mounted ARX X5 arms. The task instruction and current cameras identify the actual objects, receptacles and any conveyor; do not assume every layout has a conveyor.',
        'Both arms have parallel jaws. Use the supplied mounting transforms, jaw geometry, grasp-point convention and reach notes. For a coin or thin item, the jaw/table clearance matters; for transport, the carried object extends the swept volume.'),
    'behavior_1k': (
        'A furnished household with rooms, doors, storage fixtures and task-specific movable objects. Discover the actual layout from onboard RGB-D and remember observed routes; no global map or unseen object locations are supplied.',
        'R1Pro is a mobile dual-arm robot. The base, available torso joints, arms and grippers share a changing workspace. Use the runtime robot profile for active joints and limits. Fixtures may require opening before placing an object; identify handles, free openings and support surfaces from sensors.'),
    'robocasa': (
        'A kitchen with task-dependent counters, cabinets, drawers, sink, appliances and movable kitchen objects. The current task instruction identifies which fixtures matter. Layouts and object placements vary; no hidden object coordinates or exact fixture dimensions are supplied.',
        'PandaOmron combines a mobile base, torso and one Panda arm with a gripper. Cabinet fronts and counters can obstruct the arm or carried object. A mug handle, mug opening, spout, drawer handle and vessel rim are different contact features; use visible geometry to select the feature required by the task.'),
    'robolab': (
        'A task-specific tabletop scene seen through the supplied RGB cameras, containing the named manipulanda, containers or reference objects. Infer left/right/front/back using the task convention and camera/robot frames; do not silently substitute image-left for a world or robot axis.',
        'The fixed DROID embodiment has a Franka arm and Robotiq gripper, with no base or torso motion. The commanded flange is not the fingertips; use the adapter frame description. Plan clearance for jaws and held objects, open containers and place within their visible usable area, rather than merely above a rim.'),
    'humanoid_soccer': (
        'A ground plane or declared training-pitch visualization, with one G1 humanoid, a ball and the original goal markers. Visual fence, net or turf decoration is not an extra collision or scoring constraint unless explicitly declared by this run.',
        'The humanoid has a floating base and articulated legs, torso and arms. Moving a joint does not hold the pelvis fixed. To kick, coordinate support and the striking foot with ball motion; actual foot contact and measured body state matter. The active direct/hybrid mode and any assistance are disclosed separately.'),
    'wheeledlab': (
        'The selected case defines a drift track, terrain, visual route or World-authored precision driving course. Use that case description and the allowed sensors; a review camera is not a driver camera. Do not infer hidden checkpoints, obstacle clearance or future gate movement from privileged state.',
        'The car has wheel/steering actuators, not an omnidirectional base. Forward/reverse speed, steering sign, acceleration and slip depend on its active model. The whole vehicle footprint and rear swing matter near painted boundaries or parked cars; the centre point alone is insufficient.'),
    'ai_cps': (
        'The original single-arm precision task places a Franka Panda and its task implement in a fixed workspace. The active task is ball catching, tray balancing, peg insertion or the explicitly authored contact-recovery variant; these are separate objectives.',
        'Seven arm joints are controllable; the native task owns finger targets. The cup opening, tray support surface and peg tip have different contact geometry. A joint target is not a Cartesian tool pose. Use the task-specific measured transforms and observation fields; no additional object ground truth is supplied.'),
}

NATIVE_SCENES = {
    'T05-single': ('One Iris quadrotor and ball in the original SingleJuggleVolleyball court, without an opponent.', 'Keep the ball airborne through native body contact while stabilizing all four rotors. This is independent of T05 1v1, with its own original hit/height metrics.'),
    'T01': ('G1 walks on the native ground while carrying a free tray and a cylindrical payload; original body and object disturbances remain active. Public nominal tray size is0.254x0.352x0.018m; cylinder radius0.03m,height0.10m. Original cylinder scale randomization remains active, so these are nominal dimensions, not measurements of this episode.',
            'Legs support a floating torso while the arms support the tray. Robot, tray and payload can move relative to each other: holding hand joint angles is not equivalent to holding the payload level. There is no grasp/weld or pretrained walking policy added by this adapter.'),
    'T02': ('A Booster T1 humanoid faces a 2.74 by 1.525 m table-tennis table with a 0.76 m surface, paddle and incoming balls.',
            'Whole-body joint control must both support the robot and position/orient the paddle. The racket contact normal and contact timing determine the return; touching the ball alone is not a valid opponent-table return.'),
    'T03': ('A fixed Franka arm carries a cup while a ball is launched into the workspace. The original cup-following and virtual capture mechanism is retained.',
            'The controllable object is the arm/EEF, not the ball. Preserve cup orientation and use the reported root-frame intercept feedback. A cup-following task is not a general grasp-and-lift task; the supplied native geometry and capture rules take precedence over assumptions about a real free cup.'),
    'T04': ('The intended upstream task couples an aerial platform, suspended beam and ball. This entry is not evidence that its missing author assets have been resolved.',
            'Use only the original instantiated task contract; no substitute beam or asset geometry may silently stand in for the missing native models.'),
    'T05': ('The original Volleyball1v1 court contains two Iris drones, a ball, net and court boundaries. Each aircraft has a physical ball-contact surface.',
            'Only player 0 is controlled by your tools; the other player must have an independently declared controller. Four rotor actions act together; thrust direction changes with body attitude. A single-drone juggling variant must be named separately and cannot be scored as a 1v1 win.'),
    'T07': ('A two-wheel legged robot begins in native near-fall configurations on flat ground, with the original random disturbances.',
            'The two wheels provide a narrow support arrangement and require active balance. Virtual leg angle and length change body geometry; wheel velocity changes support motion. Native VMC tracks these targets but is not an autonomous balance controller.'),
    'T08': ('The same wheel-legged embodiment moves over original slopes, stairs and rough terrain while following sampled commands.',
            'Leg extension and wheel motion must be coordinated with actual contact and tilt. The reactive actor has no terrain scan; do not act as if a hidden map or exact upcoming step profile were provided.'),
    'T09': ('A wheeled quadruped balances on its rear wheels on the native ground under original pushes.',
            'This is a raised-body balancing configuration, not an ordinary four-wheel supported car. Wheel velocity and leg targets affect both motion and pitch stability. Coordinated feedback is required; zero action is not a supplied balancing policy.'),
    'T10': ('Unitree Go2 follows sampled body velocity/yaw commands on native flat ground, with native force disturbances when their event schedule triggers.',
            'Four legs and 12 joint targets support a floating body. Lifting feet changes support; the motor PD only follows joint angles and cannot guarantee balance. Match the runtime joint ordering, joint signs and observation scaling before coordinating a gait.'),
    'T11': ('Unitree A1 performs the native front-foot-support task on flat ground: rear feet are raised, with commanded motion retained.',
            'Front/rear refers to robot anatomy, not screen direction. Rear legs must lift while front legs support the tilted torso. The native target is not normal four-foot standing, and no handstand policy is supplied.'),
    'T13': ('Digit walks on the original ground and tracks two changing hand pose targets. There is no carried box in this task.',
            'A floating biped coordinates legs and arms. Hand target markers represent commands, not objects to grasp. Preserve body support while following hand and locomotion targets; holding the hands alone does not satisfy the walking task.'),
    'T14': ('ANYmal follows body motion commands over the original rough terrain and receives the native scheduled pushes.',
            'All 12 native joint offsets are controlled directly. Terrain heights are provided only because this actor observation includes them; do not add an external map or learned gait. Leg support, body attitude and forward tracking must be coordinated.'),
    'T15': ('The original Flamingo wheel-legged robot is intended to track motion and jump commands on its native ground. Current initialization blockers must be resolved before treating this task as runnable.',
            'Wheel and leg motor-space targets have different units and gear signs. No pretrained jump or landing controller is supplied; exact gear mapping is in the task action guide.'),
    'T16': ('A Hummingbird quadrotor carries a payload below it on a 1 m suspension, with the original payload disturbances and target.',
            'The aircraft and payload are a coupled system: stabilizing drone attitude does not by itself suppress payload swing or hold the payload target. Four raw rotor commands provide thrust and attitude moments; no hover controller is added.'),
    'T17': ('A Hummingbird supports a 1 m inverted pendulum whose tip must follow the native trajectory.',
            'The payload is above the aircraft and can topple. Track both tip motion and aircraft attitude; the native future reference samples are task commands, not permission to query hidden simulator state.'),
}


NATIVE_SCENES.update({str(task): (
    'Bench2Dex dexterous tabletop manipulation; follow the specific original task instruction and inspect the supplied native RGB cameras for props, handles, receptacles and contact surfaces.',
    'Two fixed-base UR5 arms with Wuji dexterous hands, 52 active joints in the default profile. Arm and finger targets are simultaneous absolute radians; public URDF geometry is separate from measured joint angles. There is no binary gripper, mobile base or automatic grasp planner.'
) for task in range(41, 50)})


def operating_brief(benchmark, task=None):
    scene, embodiment = NATIVE_SCENES[task] if task in NATIVE_SCENES else SCENES[benchmark]
    return '\n'.join([
        'PUBLIC OPERATING BRIEF (shared RoboDojo-style information structure)',
        'Scene and surroundings: ' + scene,
        'Robot and interaction geometry: ' + embodiment,
        'Frames, units and control strength: the task-specific instructions, exact runtime action map and tool schema below/above are authoritative. Never transfer another benchmark\'s gripper polarity, quaternion order, normalized gain or world/root frame. A command range is not a collision-free or balance-safe region.',
        'Available evidence: distinguish public model constants from current measured state and from unknown quantities. Exact object sizes, poses, contact forces, mass/friction randomization and hidden locations are unknown unless explicitly supplied by the allowed observation contract. Infer visible geometry with uncertainty; do not invent measurements or read simulator assets for answers.',
        'Feedback and contact: reason about the complete moving robot and any carried object, not just a target point. Use measured response to check signs and scale, shorten segments near contact or instability, and reassess after errors. A joint servo or IK solver is not automatically a balance controller or collision-free planner. Use only tools actually listed; a coding tool, when listed, returns the same native actions and gains no extra sensors.',
        'Completion and termination: follow the original objective and evaluator. Successful tool execution is not task success. Continue while active, but native failure and timeout remain terminal; no recovery override, reset or extension is authorized. Observation history samples tool feedback, while review video records control steps independently. Physics is paused during model deliberation.',
    ])
