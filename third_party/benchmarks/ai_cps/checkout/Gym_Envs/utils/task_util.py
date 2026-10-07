

def initialize_task(config, env, init_sim=True):

    from Tasks.Franka_Door_Open import FrankaDoorOpenTask
    from Tasks.Franka_Cloth_Placing import FrankaClothPlacingTask
    from Tasks.Franka_Cube_Stacking import FrankaCubeStackingTask
    from Tasks.Franka_Ball_Pushing import FrankaBallPushingTask
    from Tasks.Franka_Ball_Balancing import FrankaBallBalancingTask
    from Tasks.Franka_Ball_Catching import FrankaBallCatchingTask
    from Tasks.Franka_Peg_In_Hole import FrankaPegInHoleTask
    from Tasks.Franka_Point_Reaching import FrankaPointReachingTask
    
    # Mappings from strings to environments
    task_map = {
        "FrankaDoorOpen": FrankaDoorOpenTask,
        "FrankaBallPushing": FrankaBallPushingTask,
        "FrankaBallBalancing": FrankaBallBalancingTask,
        "FrankaBallCatching": FrankaBallCatchingTask,
        "FrankaPegInHole": FrankaPegInHoleTask,
        "FrankaClothPlacing": FrankaClothPlacingTask,
        "FrankaCubeStacking": FrankaCubeStackingTask,
        "FrankaPointReaching": FrankaPointReachingTask,
    }

    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig

    sim_config = SimConfig(config)

    cfg = sim_config.config
    task = task_map[cfg["task_name"]](
        name=cfg["task_name"], sim_config=sim_config, env=env
    )

    env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=init_sim)

    return task
