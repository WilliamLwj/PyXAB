# -*- coding: utf-8 -*-
"""Implementation of GPO (Shang et al., 2019)
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT

import numpy as np
from PyXAB.algos.Algo import Algorithm


class GPO(Algorithm):
    """
    Implementation of the General Parallel Optimization (GPO) algorithm (Shang et al., 2019)
    """

    def __init__(
        self, numax=1.0, rhomax=0.9, rounds=1000, domain=None, partition=None, algo=None
    ):
        """
        Initialization of the wrapper algorithm

        Parameters
        ----------
        numax: float
            parameter nu_max in the algorithm (default 1.0)
        rhomax: float
            parameter rho_max in the algorithm, the maximum rho used (default 0.9)
        rounds: int
            the number of rounds/budget (default 1000)
        domain: list(list)
            the domain of the objective function
        partition:
            the partition used in the optimization process
        algo:
            the baseline algorithm used by the wrapper, such as T_HOO or HCT
        """
        super(GPO, self).__init__()
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        if algo is None:
            raise ValueError("Algorithm for GPO is not given")

        self.rounds = rounds
        self.rhomax = rhomax
        self.numax = numax
        self.Dmax = np.log(2) / np.log(1 / rhomax)
        self.domain = domain
        self.partition = partition
        self.algo = algo

        # The big-N in the algorithm
        self.N = np.ceil(
            0.5 * self.Dmax * np.log(self.rounds / 2) / np.log(self.rounds / 2)
        )

        # phase number
        self.phase = 1

        # Starts with a none algorithm
        self.curr_algo = None
        self.half_phase_length = np.floor(self.rounds / (2 * self.N))
        self.counter = 0
        self.goodx = None

        # The cross-validation list
        self.V_x = []
        self.V_reward = []

    def pull(self, time):
        """
        The pull function of GPO that returns a point to be evaluated

        Parameters
        ----------
        time: int
            The time step of the online process.

        Returns
        -------
        point: list
            The point chosen by the GPO algorithm
        """
        if self.phase > self.N:  # If already finished
            return self.goodx
        else:
            if self.counter == 0:
                rho = self.rhomax ** (2 * self.N / (2 * self.phase + 1))
                # TODO: for algorithms that do not need nu or rho
                self.curr_algo = self.algo(
                    nu=self.numax, rho=rho, domain=self.domain, partition=self.partition
                )

            if self.counter < self.half_phase_length:
                point = self.curr_algo.pull(time)
                self.goodx = point

            elif self.counter == self.half_phase_length:
                point = self.goodx
                self.V_x.append(point)
                self.V_reward.append(0)
            else:
                point = self.goodx

            if self.counter >= 2 * self.half_phase_length:
                self.phase += 1
                self.counter = 0

        return point

    def receive_reward(self, time, reward):
        """
        The receive_reward function of GPO to receive the reward for the chosen point

        Parameters
        ----------
        time: int
            The time step of the online process.

        reward: float
            The (Stochastic) reward of the pulled point

        Returns
        -------
        """
        if self.phase > self.N:  # If already finished
            pass
        elif self.phase == self.N:
            maxind = np.argmax(np.array(self.V_reward))
            self.goodx = self.V_x[maxind]

        else:
            if self.counter < self.half_phase_length:
                self.curr_algo.receive_reward(time, reward)
            else:
                self.V_reward[self.phase - 1] = (
                    self.V_reward[self.phase - 1]
                    * (self.counter - self.half_phase_length)
                    + reward
                ) / (self.counter - self.half_phase_length + 1)

        self.counter += 1
