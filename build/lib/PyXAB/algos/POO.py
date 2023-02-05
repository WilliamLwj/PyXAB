# -*- coding: utf-8 -*-
"""Implementation of GPO (Shang et al, 2019)
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT

import numpy as np
from PyXAB.algos.Algo import Algorithm


class POO(Algorithm):

    def __init__(self, numax=1, rhomax=0.9, rounds=1000, domain=None, partition=None, algo=None):
        super(POO, self).__init__()
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        if algo is None:
            raise ValueError("Algorithm for POO is not given")

        self.rounds = rounds
        self.rhomax = rhomax
        self.numax = numax
        self.Dmax = np.log(2) / np.log(1/rhomax) #TODO: Change this 2 to K-arm
        self.domain= domain
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

    def pull(self, time):


        if self.N <= 0.5 * self.Dmax * np.log(self.n/np.log(self.n)):
            if self.counter == 0:
                rho = self.rhomax ** (2 * self.N / (2 * self.phase + 1))
                self.curr_algo = self.algo(nu=self.numax, rho=rho, domain=self.domain, partition=self.partition)
                self.V_algo.append(self.curr_algo)
                self.V_reward.append(0)
            point = self.curr_algo.pull(time)

            if self.counter >= np.ceil(self.N / self.n):
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

            algo = self.V_algo[self.algo_counter]
            point = algo.pull(time)
            self.algo_counter += 1
            if self.algo_counter == len(self.V_algo):
                self.algo_counter = 0
                self.n = self.n + self.N

        return point



    def receive_reward(self, time, reward):

        if self.N <= 0.5 * self.Dmax * np.log(self.n/np.log(self.n)):
            self.curr_algo.receive_reward(time, reward)
            self.V_reward[-1] = (self.V_reward[-1] * (self.counter) + reward) / (self.counter + 1)
            self.counter += 1

        else:
            self.V_algo[self.algo_counter].receive_reward(time, reward)
            self.V_reward[self.algo_counter] = (self.V_reward[self.algo_counter] * np.ceil(self.N / self.n) + reward) \
                    /(np.ceil(self.N / self.n) + 1)

    def get_last_point(self):

        V_reward = np.array(self.V_reward)

        max_param = np.argmax(V_reward)

        point = self.V_algo[max_param].pull()

        return point