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
    opt_type: type of the optimizer
    budget_size: local budget size
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

        # initial value bounds
        if self.task_name is "FrankaBallPushing":

            self.bnds = ((-0.1, 0.1), (-0.1, 0.1))

        elif self.task_name is "FrankaBallBalancing":

            self.bnds = ((-0.15, 0.15), (-0.15, 0.15))

        elif self.task_name is "FrankaBallCatching":

            # self.bnds = ((-0.1, 0.1), (-0.2, 0.2), (1.0, 3.0), (-1.0, 1.0))
            self.bnds = ((-0.05, 0.05), (-0.05, 0.05), (1.0, 1.001), (0.0, 0.001))

        elif self.task_name is "FrankaCubeStacking":

            self.bnds = ((-0.2, 0.2), (-0.2, 0.2))

        elif self.task_name is "FrankaDoorOpen":

            self.bnds = ((-0.025, 0.025), (-0.05, 0.05))

        elif self.task_name is "FrankaPegInHole":

            self.bnds = ((-0.1, 0.1), (-0.1, 0.1))

        elif self.task_name is "FrankaPointReaching":

            self.bnds = ((-0.2, 0.2), (-0.4, 0.4), (-0.2, 0.2))

        elif self.task_name is "FrankaClothPlacing":

            self.bnds = ((-0.1, 0.2), (-0.35, 0.35))

        else:

            raise ValueError("Task name unknown for generating the initial values")

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

        return min_rob

    # optimization based on the optimizer type
    def optimize(self):

        if self.opt_type is "random":

            results = self.optimize_random()

            return results

        elif self.opt_type is "NelderMead":

            results = self.optimize_nelder_mead()

            return results

        elif self.opt_type is "DualAnnealing":

            results = self.optimize_dual_annealing()

            return results

        else:

            raise ValueError("Optimizer type undefined!")

    # Random optimization
    def optimize_random(self):

        # worst_initial = None
        # worst_trace = None

        initial_value_record = None
        rob_value_record = None

        # Random optimizer
        for i in range(self.budget_size):

            # random initial value
            initial_value = self.generate_initial()
            # compute robustness
            min_rob = self.robustness_function(initial_value)

            # update record
            if i == 0:

                initial_value_record = initial_value
                rob_value_record = np.array([min_rob])

                # worst_initial = initial_value
                # worst_trace = trace
                self.worst_rob = min_rob

            else:

                initial_value_record = np.vstack((initial_value_record, initial_value))
                rob_value_record = np.vstack((rob_value_record, np.array([min_rob])))

                if min_rob < self.worst_rob:

                    # worst_initial = initial_value
                    # worst_trace = trace
                    self.worst_rob = min_rob

            if min_rob < 0:
                # self.fal_succ = True
                # self.fal_time = time.time() - self.start_time
                if i == 0:
                    self.fal_sim = 1
                break

        # results = {'worst_initial': worst_initial, 'worst_rob': worst_rob,
        #            'initial_value_record': initial_value_record, 'rob_value_record': rob_value_record}

        if self.fal_succ == False:
            self.fal_time = time.time() - self.start_time

        results = [self.fal_succ, self.fal_time, self.fal_sim, self.worst_rob]

        return results

    # Nelder Mead optimization
    def optimize_nelder_mead(self):

        initial_guess = self.generate_initial()

        # minimization
        results = minimize(
            self.robustness_function,
            initial_guess,
            method="Nelder-Mead",
            bounds=self.bnds,
            options={"maxfev": self.budget_size, "disp": True},
        )

        if self.fal_succ == False:
            self.fal_time = time.time() - self.start_time

        results = [self.fal_succ, self.fal_time, self.fal_sim, self.worst_rob]

        return results

    # Dual Annealing optimization
    def optimize_dual_annealing(self):

        # minimization
        results = dual_annealing(
            self.robustness_function,
            bounds=self.bnds,
            # maxiter=self.budget_size, # global search number
            maxfun=self.budget_size,  # local search number
            # no_local_search = True,
        )

        if self.fal_succ == False:
            self.fal_time = time.time() - self.start_time

        results = [self.fal_succ, self.fal_time, self.fal_sim, self.worst_rob]

        return results
