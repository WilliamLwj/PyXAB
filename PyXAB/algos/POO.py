# -*- coding: utf-8 -*-
"""Implementation of POO (Grill et al., 2019)
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT
import pdb

import numpy as np
from PyXAB.algos.Algo import Algorithm


class POO(Algorithm):
    """
    Implementation of the Parallel Optimistic Optimization (POO) algorithm (Grill et al., 2015), with the general
    definition in Shang et al., 2019.
    """

    def __init__(
        self, numax=1, rhomax=0.9, rounds=1000, domain=None, partition=None, algo=None
    ):
        """

        Parameters
        ----------
        numax: float
            parameter nu_max in the algorithm
        rhomax: float
            parameter rho_max in the algorithm, the maximum rho used
        rounds: int
            the number of rounds/budget
        domain: list(list)
            the domain of the objective function
        partition:
            the partition used in the optimization process
        algo:
            the baseline algorithm used by the wrapper, such as T_HOO or HCT
        """
        super(POO, self).__init__()
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        if algo is None:
            raise ValueError("Algorithm for POO is not given")
        if (
            algo.__name__ != "T_HOO"
            and algo.__name__ != "HCT"
            and algo.__name__ != "VHCT"
        ):
            raise NotImplementedError(
                "POO has not yet included implementations for this algorithm"
            )

        self.rounds = rounds
        self.rhomax = rhomax
        self.numax = numax
        self.Dmax = np.log(2) / np.log(1 / rhomax)  # TODO: Change this 2 to K-arm
        self.domain = domain
        self.partition = partition
        self.algo = algo

        # The big-N and small-n in the algorithm (The first iteration is useless)
        self.N = 2
        self.n = self.N

        # phase number
        self.phase = 1

        # Starts with a none algorithm
        self.curr_algo = None
        self.counter = 0
        self.goodx = None

        # The cross-validation list
        self.V_algo = []
        self.V_reward = []
        self.Times = []

    def pull(self, time):
        """
        The pull function of POO that returns a point to be evaluated

        Parameters
        ----------
        time: int
            The time step of the online process.

        Returns
        -------
        point: list
            The point chosen by the POO algorithm
        """

        if self.N <= 0.5 * self.Dmax * np.log(self.n / np.log(self.n)):

            if self.counter == 0:
                rho = self.rhomax ** (2 * self.N / (2 * self.phase + 1))
                if self.algo.__name__ == "T_HOO":
                    self.curr_algo = self.algo(
                        nu=self.numax,
                        rho=rho,
                        rounds=self.rounds,
                        domain=self.domain,
                        partition=self.partition,
                    )
                elif self.algo.__name__ == "HCT" or self.algo.__name__ == "VHCT":
                    self.curr_algo = self.algo(
                        nu=self.numax,
                        rho=rho,
                        domain=self.domain,
                        partition=self.partition,
                    )
                # TODO: add more algorithms that do not need nu or rho
                self.V_algo.append(self.curr_algo)
                self.V_reward.append(0)
                self.Times.append(0)
            point = self.V_algo[-1].pull(time)

        else:
            algo = self.V_algo[self.algo_counter]
            point = algo.pull(time)

        return point

    def receive_reward(self, time, reward):
        """
        The receive_reward function of POO to receive the reward for the chosen point

        Parameters
        ----------
        time: int
            The time step of the online process.

        reward: float
            The (Stochastic) reward of the pulled point

        Returns
        -------

        """
        if self.N <= 0.5 * self.Dmax * np.log(self.n / np.log(self.n)):
            self.V_algo[-1].receive_reward(time, reward)
            self.V_reward[-1] = (self.V_reward[-1] * self.counter + reward) / (
                self.counter + 1
            )
            self.Times[-1] += 1
            self.counter += 1
            if self.counter >= np.ceil(self.n / self.N):
                self.counter = 0
                self.phase += 1

            # Refresh, change n and N
            if self.phase >= self.N:
                self.n = 2 * self.n
                self.N = 2 * self.N
                self.phase = 0
                self.counter = 0
                self.algo_counter = 0
        else:
            self.V_algo[self.algo_counter].receive_reward(time, reward)
            self.V_reward[self.algo_counter] = (
                self.V_reward[self.algo_counter] * np.ceil(self.n / self.N) + reward
            ) / (np.ceil(self.n / self.N) + 1)
            self.Times[self.algo_counter] += 1
            self.algo_counter += 1
            if self.algo_counter == len(self.V_algo):
                self.algo_counter = 0
                self.n = self.n + self.N

    def get_last_point(self):
        """
        The function that returns the last point chosen by POO

        Returns
        -------

        """
        V_reward = np.array(self.V_reward)
        max_param = np.argmax(V_reward)
        point = self.V_algo[max_param].pull(time=0)
        return point
