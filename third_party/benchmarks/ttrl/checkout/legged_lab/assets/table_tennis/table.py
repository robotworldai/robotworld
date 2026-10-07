import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets.rigid_object import RigidObjectCfg

from legged_lab.assets import ISAAC_ASSET_DIR

 #NOTE: The origin of the table frame is at the ground surface and the z-axis passes through the center of the table top surface.
# - Collision properties : Convex decomposition (contact offset: 0.001m, rest offset:0.0005m)
# - Table geometry (X-Y-Z):  (2.74, 1.525, 0.76)m

TABLE_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Table",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/table_tennis/table/pp_table_ver2.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0025,
            max_depenetration_velocity=1000.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), 
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel = (0.0, 0.0, 0.0),
        ang_vel = (0.0, 0.0, 0.0),

    ),
)