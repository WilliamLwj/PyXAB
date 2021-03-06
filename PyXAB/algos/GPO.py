# -*- coding: utf-8 -*-
"""Implementation of GPO (Shang et al, 2019)
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT

import numpy as np
from PyXAB.algos.Algo import Algorithm


class GPO(Algorithm):

    def __init__(self, numax=1, rhomax=0.9, rounds=1000, domain=None, partition=None, algo=None):
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
        self.Dmax = np.log(2) / np.log(1/rhomax)
        self.domain= domain
        self.partition = partition
        self.algo = algo


        # The big-N in the algorithm
        self.N = np.ceil(0.5 * self.Dmax * np.log(self.rounds/2) / np.log(self.rounds/2))

        # phase number
        self.phase = 1

        # Starts with a none algorithm
        self.curr_algo = None
        self.half_phase_length = np.floor(self.rounds/(2 * self.N))
        self.counter = 0
        self.goodx = None


        # The cross-validation list
        self.V_x = []
        self.V_reward = []

    def pull(self, time):

        if self.phase > self.N: # If already finished
            return self.goodx
        else:
            if self.counter == 0:
                rho = self.rhomax ** (2 * self.N / (2 * self.phase + 1))
                # TODO: for algorithms that do not need nu or rho
                self.curr_algo = self.algo(nu=self.numax, rho=rho, domain=self.domain, partition=self.partition)

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

        if self.phase > self.N: # If already finished
            pass
        elif self.phase == self.N:
            maxind = np.argmax(np.array(self.V_reward))
            self.goodx = self.V_x[maxind]

        else:
            if self.counter < self.half_phase_length:
                self.curr_algo.receive_reward(time, reward)
            else:
                self.V_reward[self.phase-1] = (self.V_reward[self.phase-1] * (self.counter - self.half_phase_length) + reward) \
                                              / (self.counter - self.half_phase_length + 1)


        self.counter += 1