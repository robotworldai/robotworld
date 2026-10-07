from typing import Optional

import sys
import numpy as np
import torch
import time

from scipy.optimize import minimize
from scipy.optimize import dual_annealing


class Optimizer(object):
    """Optimizer class for testing
    task_name: the task name of environment
    test_model: the model under test
    monitor: the monitor for the STL specification
    """

    def __init__(
        self,
        task_name,
        test_model,
        monitor,
        opt_type: Optional[str] = "random",
        budget_size: Optional[int] = 1000,
    ):

        self.task_name = task_name
        self.test_model = test_model
        self.monitor = monitor
        self.opt_type = opt_type
        self.budget_size = budget_size
        self.fal_succ = False
        self.start_time = time.time()
        self.fal_time = 0
        self.fal_sim = 0
        self.worst_rob = 1000

    # generate initial values based on the task type
    def generate_initial(self):

        if self.task_name is "FrankaBallPushing":

            # ball inside an area x:[-0.1,0.1], y:[-0.1,0.1]
            value_1 = np.random.rand(1) * (0.1 + 0.1) - 0.1
            value_2 = np.random.rand(1) * (0.1 + 0.1) - 0.1
            initial_value = np.hstack((value_1, value_2))

        elif self.task_name is "FrankaBallBalancing":

            # ball inside an area x:[-0.15,0.15], y:[-0.15,0.15]
            value_1 = np.random.rand(1) * (0.15 + 0.15) - 0.15
            value_2 = np.random.rand(1) * (0.15 + 0.15) - 0.15
            initial_value = np.hstack((value_1, value_2))

        elif self.task_name is "FrankaBallCatching":

            # ball inside an area x:[-0.1,0.1], y:[-0.1,0.1]
            # ball velociry: vx: [1.0,1.5], vy: [0.0,0.2]
            value_1 = np.random.rand(1) * (0.05 + 0.05) - 0.05
            value_2 = np.random.rand(1) * (0.05 + 0.05) - 0.05
            value_3 = np.random.rand(1) * (1.0 - 1.0) + 1.0
            value_4 = np.random.rand(1) * (0.0 + 0.0) + 0.0
            initial_value = np.hstack((value_1, value_2, value_3, value_4))

        elif self.task_name is "FrankaCubeStacking":

            # target cube inside an area x:[-0.2,0.2], y:[-0.2,0.2]
            value_1 = np.random.rand(1) * (0.2 + 0.2) - 0.2
            value_2 = np.random.rand(1) * (0.2 + 0.2) - 0.2
            initial_value = np.hstack((value_1, value_2))

        elif self.task_name is "FrankaDoorOpen":

            # target inside an area x:[-0.1,0.1], y:[-0.4,0.4]
            value_1 = np.random.rand(1) * (0.005 + 0.005) - 0.005
            value_2 = np.random.rand(1) * (0.025 + 0.025) - 0.025
            initial_value = np.hstack((value_1, value_2))

        elif self.task_name is "FrankaPegInHole":

            # target inside an area x:[-0.2,0.2], y:[-0.2,0.2]
            value_1 = np.random.rand(1) * (0.1 + 0.1) - 0.1
            value_2 = np.random.rand(1) * (0.1 + 0.1) - 0.1
            initial_value = np.hstack((value_1, value_2))

        elif self.task_name is "FrankaPointReaching":

            # target inside an area x:[-0.2,0.2], y:[-0.4,0.4], z:[-0.2,0.2]
            value_1 = np.random.rand(1) * (0.2 + 0.2) - 0.2
            value_2 = np.random.rand(1) * (0.4 + 0.4) - 0.4
            value_3 = np.random.rand(1) * (0.2 + 0.2) - 0.2
            initial_value = np.hstack((value_1, value_2, value_3))

        elif self.task_name is "FrankaClothPlacing":

            # target inside an area x:[-0.1,0.2], y:[-0.35,0.35]
            value_1 = np.random.rand(1) * (0.2 + 0.1) - 0.1
            value_2 = np.random.rand(1) * (0.35 + 0.35) - 0.35
            initial_value = np.hstack((value_1, value_2))

        else:

            raise ValueError("Task name unknown for generating the initial values")

        return initial_value

    # Generate one function (input: initial values, output: robustness) for testing algorithms
    def robustness_function(self, initial_value):

        # print("Initial Value:", initial_value)
        # Get trace
        trace = self.test_model.compute_trace(initial_value)
        indexed_trace = self.test_model.merge_trace(trace)

        # compute robustness
        rob_sequence = self.monitor.compute_robustness(indexed_trace)
        rob_sequence = np.array(rob_sequence)

        # RTAMT is for monitoring, so for eventually, the robustness computed from the current timepoint to the end
        # workaround to compute the maximum
        if (
            self.task_name is "FrankaBallPushing"
            or self.task_name is "FrankaCubeStacking"
            or self.task_name is "FrankaDoorOpen"
            or self.task_name is "FrankaPegInHole"
            or self.task_name is "FrankaClothPlacing"
        ):
            min_rob = np.max(rob_sequence[:, 1])
        else:
            min_rob = np.min(rob_sequence[:, 1])

        # print("Min Robustness:", min_rob)

        if min_rob < self.worst_rob:
            self.worst_rob = min_rob

        if min_rob < 0 and self.fal_succ == False:
            self.fal_succ = True
            self.fal_time = time.time() - self.start_time
        elif self.fal_succ == False:
            self.fal_sim += 1

        return min_rob, rob_sequence, indexed_trace

    # optimization based on the optimizer type
    def optimize(self):

        if self.opt_type is "random":

            results = self.optimize_random()

            return results

        else:

            raise ValueError("Optimizer type undefined!")

    # Random optimization
    def optimize_random(self):

        success_count = 0           # num success trail/ num total trail
        dangerous_rate = list()     # num dangerous steps/ num total trail w.r.t each trail
        completion_time = list()    # the step that indicates the task is completed

        # Random optimizer
        for i in range(self.budget_size):

            print("trail ",i)

            # random initial value
            initial_value = self.generate_initial()
            # compute robustness and its sequence
            min_rob, rob_sequence, indexed_trace = self.robustness_function(initial_value)

            # compute dangerous_rate, completion_time w.r.t tasks
            if self.task_name == "FrankaCubeStacking":
                # info extraction
                cube_dist = np.array(indexed_trace["distance_cube"])[:,1]
                cube_z_dist = np.array(indexed_trace["z_cube_distance"])[:,1]
                # dangerous rate:
                cube_too_far = cube_dist >= 0.35
                cube_fall_ground = cube_z_dist < 0.02
                dangerous_rate.append(np.sum(np.logical_or(cube_too_far, cube_fall_ground))/len(cube_dist))
                # completation step
                if_complete = (np.logical_and(cube_dist<=0.024, cube_z_dist>0))
                complete_Step = np.where(if_complete == True)[0]
                if len(complete_Step) > 0: 
                    completion_time.append(complete_Step[0])

            elif self.task_name == "FrankaDoorOpen":
                handle_yaw = np.array(indexed_trace)[:,1]
                # dangerous rate:
                dangerous_rate.append(np.sum(handle_yaw<0.1)/len(handle_yaw))
                # completation step
                if_complete = (handle_yaw>=20)
                complete_Step = np.where(if_complete == True)[0]
                if len(complete_Step) > 0: 
                    completion_time.append(complete_Step[0])

            elif self.task_name == "FrankaPegInHole":
                tool_hole_distance = np.array(indexed_trace)[:,1]
                # dangerous rate:
                dangerous_rate.append(np.sum(tool_hole_distance>0.37)/len(tool_hole_distance))
                # completation step
                if_complete = (tool_hole_distance<=0.1)
                complete_Step = np.where(if_complete == True)[0]
                if len(complete_Step) > 0: 
                    completion_time.append(complete_Step[0])

            elif self.task_name == "FrankaBallCatching":
                ball_tool_distance = np.array(indexed_trace)[:,1]
                # dangerous rate:
                dangerous_rate.append(np.sum(ball_tool_distance>0.2)/len(ball_tool_distance))
                # completation step
                if_complete = (ball_tool_distance<=0.1)
                complete_interval = np.zeros(len(if_complete)-5)
                # spec satisified holds within a 3-step interval
                for i in range(0, int(len(if_complete)-5)):
                    complete_interval[i] = np.all(if_complete[i:i+5])
                complete_Step = np.where(complete_interval == True)[0]
                if len(complete_Step) > 0: 
                    completion_time.append(complete_Step[0])

            elif self.task_name == "FrankaBallBalancing":
                ball_tool_distance = np.array(indexed_trace)[:,1]
                # dangerous rate:
                dangerous_rate.append(np.sum(ball_tool_distance>0.2)/len(ball_tool_distance))
                # completation step
                if_complete = (ball_tool_distance<=0.1)
                complete_interval = np.zeros(len(if_complete)-5)
                # spec satisified holds within a 3-step interval
                for i in range(0, int(len(if_complete)-5)):
                    complete_interval[i] = np.all(if_complete[i:i+5])
                complete_Step = np.where(complete_interval == True)[0]
                if len(complete_Step) > 0: 
                    completion_time.append(complete_Step[0])

            elif self.task_name == "FrankaBallPushing":
                ball_hole_distance = np.array(indexed_trace)[:,1]
                # dangerous rate:
                dangerous_rate.append(np.sum(ball_hole_distance>0.5)/len(ball_hole_distance))
                # completation step
                if_complete = (ball_hole_distance<=0.3)
                complete_interval = np.zeros(len(if_complete)-5)
                # spec satisified holds within a 3-step interval
                for i in range(0, int(len(if_complete)-5)):
                    complete_interval[i] = np.all(if_complete[i:i+5])
                complete_Step = np.where(complete_interval == True)[0]
                if len(complete_Step) > 0: 
                    completion_time.append(complete_Step[0])

            elif self.task_name == "FrankaPointReaching":
                finger_target_distance = np.array(indexed_trace)[:,1]
                # dangerous rate:
                dangerous_rate.append(np.sum(finger_target_distance>=0.6)/len(finger_target_distance))
                # completation step
                if_complete = (finger_target_distance<=0.12)
                complete_Step = np.where(if_complete == True)[0]
                if len(complete_Step) > 0: 
                    completion_time.append(complete_Step[0])

            elif self.task_name == "FrankaClothPlacing":
                # info extraction
                cloth_target_dist = np.array(indexed_trace["distance_cloth_target"])[:,1]
                cloth_z_pos = np.array(indexed_trace["cloth_height"])[:,1]
                # dangerous rate:
                cloth_too_far = cloth_target_dist >= 0.3
                cloth_fall_ground = cloth_z_pos < 0.1
                dangerous_rate.append(np.sum(np.logical_or(cloth_too_far, cloth_fall_ground))/len(cloth_target_dist))
                # completation step
                if_complete = cloth_target_dist<=0.25
                complete_Step = np.where(if_complete == True)[0]
                if len(complete_Step) > 0: 
                    completion_time.append(complete_Step[0])
                # print(indexed_trace)
            else:
                print("Invalid Task")
                break

            # perform evaluation:
            # success rate
            if min_rob > 0:
                success_count += 1

            # dangerous behavior: change the STL specification and use rob_sequence?

            # completion time: check first satisfication in index_trace?

            # if i ==  0:
            #     break
            
        if len(dangerous_rate) == 0:
            dangerous_rate = 0

        results = {"success_count": success_count/self.budget_size, 
                    "dangerous_rate": np.mean(dangerous_rate),
                    "completion_time": np.mean(completion_time)}

        return results