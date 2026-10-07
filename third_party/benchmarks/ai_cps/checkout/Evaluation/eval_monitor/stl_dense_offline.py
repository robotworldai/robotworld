from rtamt import STLDenseTimeSpecification
from typing import Optional

import sys


class stl_dense_offline_monitor(object):
    """STL dense time offline monitor based rtamt
    agent_path: the path to the agent parameters (checkpoint)
    oige_path: path to the OIGE environment;
    agent_type: type of DRL agent (PPO, DDPG, TRPO)
    task_name: the name of the task
    num_envs: the number of parallel running environments
    """

    def __init__(
        self,
        task_name: Optional[str] = None,
        agent_type: Optional[str] = None,
        oige_path: Optional[str] = None,
    ):

        if task_name is not None:
            self.task_name = task_name
        else:
            self.task_name = "FrankaBallPushing"

        self.agent_type = agent_type

        self.generate_spec()

    # generate specification based on task name
    def generate_spec(self):

        # Initialization
        self.spec = STLDenseTimeSpecification()
        self.spec.name = "STL Dense-time Offline Monitor"

        ###############################################
        # Specification according to task

        # Ball Pushing
        if self.task_name is "FrankaBallPushing":

            self.spec.declare_var("distance_ball_hole", "float")
            self.spec.spec = "eventually[1:299](distance_ball_hole <= 0.3) "

        # Ball Balancing
        elif self.task_name is "FrankaBallBalancing":

            self.spec.declare_var("distance_ball_tool", "float")
            self.spec.spec = "always[50:200]( distance_ball_tool <= 0.25)"

        # Ball Catching
        elif self.task_name is "FrankaBallCatching":

            self.spec.declare_var("distance_ball_tool", "float")
            self.spec.spec = "always[50:299]( distance_ball_tool <= 0.1)"

        # Cube Stacking
        elif self.task_name is "FrankaCubeStacking":

            self.spec.declare_var("distance_cube", "float")
            self.spec.declare_var("z_cube_distance", "float")
            self.spec.spec = (
                "eventually[1:299]((distance_cube<= 0.024) and (z_cube_distance>0) )"
            )

        # Door Open
        elif self.task_name is "FrankaDoorOpen":

            self.spec.declare_var("yaw_door", "float")
            self.spec.spec = "eventually[1:299]( yaw_door >= 20)"

        # Peg In Hole
        elif self.task_name is "FrankaPegInHole":

            self.spec.declare_var("distance_tool_hole", "float")
            self.spec.spec = "always[250:299]( distance_tool_hole <= 0.1)"

        # Point Reaching
        elif self.task_name is "FrankaPointReaching":

            self.spec.declare_var("distance_finger_target", "float")
            self.spec.spec = "always[50:299]( distance_finger_target <= 0.12)"  # fixed

        # Cloth Placing
        elif self.task_name is "FrankaClothPlacing":

            self.spec.declare_var("distance_cloth_target", "float")
            self.spec.declare_var("cloth_height", "float")
            self.spec.spec = "eventually[1:299]( (distance_cloth_target <= 0.25))" # and (cloth_height > 0.1) )"

        else:

            raise ValueError("Task name unknown for defining the specification")

        ################################################

        # Load specification
        try:
            self.spec.parse()
        except rtamt.STLParseException as err:
            print("STL Parse Exception: {}".format(err))
            sys.exit()

    # Compute the robustness given trace
    def compute_robustness(self, trace):

        if self.task_name is "FrankaBallPushing":

            # print(trace)
            robustness = self.spec.evaluate(["distance_ball_hole", trace])
            # print(robustness)

        elif self.task_name is "FrankaBallBalancing":

            robustness = self.spec.evaluate(["distance_ball_tool", trace])

        elif self.task_name is "FrankaBallCatching":

            robustness = self.spec.evaluate(["distance_ball_tool", trace])

        elif self.task_name is "FrankaCubeStacking":

            distance_cube = trace["distance_cube"]
            z_cube_distance = trace["z_cube_distance"]

            robustness = self.spec.evaluate(
                ["distance_cube", distance_cube], ["z_cube_distance", z_cube_distance]
            )

        elif self.task_name is "FrankaDoorOpen":

            robustness = self.spec.evaluate(["yaw_door", trace])

        elif self.task_name is "FrankaPegInHole":

            robustness = self.spec.evaluate(["distance_tool_hole", trace])

        elif self.task_name is "FrankaPointReaching":

            robustness = self.spec.evaluate(["distance_finger_target", trace])

        elif self.task_name is "FrankaClothPlacing":

            distance_cloth_target = trace["distance_cloth_target"]
            cloth_height = trace["cloth_height"]

            # print("distance")
            # print(distance_cloth_target)
            # print(cloth_height)

            robustness = self.spec.evaluate(
                ["distance_cloth_target", distance_cloth_target]#, ["cloth_height", cloth_height]
            )

            # print("rob: ")
            # print(robustness)

        else:

            raise ValueError("Task name unknown for defining the specification")

        return robustness
