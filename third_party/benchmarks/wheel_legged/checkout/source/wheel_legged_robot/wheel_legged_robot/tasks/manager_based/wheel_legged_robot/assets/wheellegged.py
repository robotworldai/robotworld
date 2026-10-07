import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from . import WHEEL_LEGGED_ASSET_DIR

##
# Configuration
##

WHEEL_LEGGED_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,                     # 必须浮动，用于平衡控制
        merge_fixed_joints=True,            # 合并固定关节（如果有）
        replace_cylinders_with_capsules=True, # 你的连杆不是标准圆柱，保持 False
        asset_path=str(WHEEL_LEGGED_ASSET_DIR / "urdf" / "wl_dealed.urdf"),
        activate_contact_sensors=True,      # 开启轮子接触检测
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,   # 避免左右腿互穿
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
        # 此处 joint_drive 的 stiffness/damping 会被 actuators 覆盖，设 0 即可
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25),
        joint_pos={
            "lf0_Joint": 0.5,      # 髋关节直立
            "rf0_Joint": -0.5,
            "lf1_Joint": 0.35,      # 膝关节微屈，保持平衡姿态
            "rf1_Joint": -0.35,
            "l_wheel_Joint": 0.0,  # 轮子初始角度无所谓
            "r_wheel_Joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # 腿部关节：使用力矩控制
        "legs": DCMotorCfg(
            joint_names_expr=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
            effort_limit=30.0,          # 根据 URDF 中的 effort
            saturation_effort=30.0,
            velocity_limit=30.0,        # rad/s
            # 全部为0,做力矩控制：
            stiffness=0.0,             # 位置增益
            damping=0.0,                # 速度增益
            friction=0.0,
        ),
        # 轮子：使用力矩控制
        "wheels": DCMotorCfg(
            joint_names_expr=["l_wheel_Joint", "r_wheel_Joint"],
            effort_limit=5.0,           # URDF 中 effort=5
            saturation_effort=5.0,
            velocity_limit=1000.0,      # 轮子允许高速
            # 全部为0,做力矩控制：
            stiffness=0.0,              # 如果做速度控制，stiffness 设 0，用 damping 作比例增益
            damping=0.0,                # 速度控制的比例系数
            friction=0.0,
        ),
    },
)
