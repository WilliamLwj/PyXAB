# -*- coding: utf-8 -*-
"""Implementation of GPO (Shang et al, 2019)
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT

import math
import numpy as np
import pdb
from algos.Algo import Algorithm


class GPO(Algorithm):

    def __init__(self, rounds=1000, rhomax=0.9, numax=1, domain=None, partition=None, algo=None):
        super(GPO, self).__init__()
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        self.partition = partition()

        self.rounds = rounds
        self.rhomax = rhomax
        self.numax = numax
        self.Dmax = np.log(2) / np.log(1/rhomax)
        self.algo = algo

    def run(self, time):

        N = int(0.5 * self.Dmax * np.log(self.rounds/2) / np.log(self.rounds/2))
        n = 0
        for i in range(1, N+1):

            rho = self.rhomax**(2*N/(2*i+1))
            tree = HCT.HCT_tree(nu_max, rho, support)

            for k in range(int(rounds/(2*N))):
                curr_node, path = tree.optTraverse()
                sample_range = curr_node.range
                pulled_x = []
                for j in range(len(sample_range)):
                    x = (sample_range[j][0] + sample_range[j][1]) / 2.0
                    pulled_x.append(x)
                reward = Target.f(pulled_x) + random.uniform(-1 * noise, noise)
                tree.updateAllTree(path, curr_node, reward)
                n += 1
                simple_regret = Target.fmax - Target.f(pulled_x)
                regret += simple_regret
                PCT_regret_list.append(regret)
                # print(n, pulled_x)
                if n > rounds:
                    break
            for k in range(int(rounds / (2 * N))):
                reward = Target.f(pulled_x) + random.uniform(-1 * noise, noise)
                tree.updateAllTree(path, curr_node, reward)
                # print(n, pulled_x)
                n += 1
                simple_regret = Target.fmax - Target.f(pulled_x)
                regret += simple_regret
                PCT_regret_list.append(regret)
                if n > rounds:
                    break

    def receive_reward(self, time, reward):

        pass