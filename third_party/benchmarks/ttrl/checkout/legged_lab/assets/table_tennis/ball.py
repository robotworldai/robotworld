import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets.rigid_object import RigidObjectCfg

from legged_lab.assets import ISAAC_ASSET_DIR

BALL_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.02,  # Table tennis ball radius (20mm)
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=8, #32
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0025,
            max_depenetration_velocity=1000.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.0034),  # ! Vicon Table tennis ball mass (3.4g)
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.001,
            rest_offset=0.0005,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.0),  # Red ball
            metallic=0.0,
            roughness=0.5,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.1,
            dynamic_friction=0.1,
            restitution=0.9,  # High bounce for table tennis ball
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(1.35, 0.0, 1.03),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    ),
)