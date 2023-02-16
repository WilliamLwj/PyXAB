# -*- coding: utf-8 -*-
"""Implementation of SequOOL (Bartlett, 2019)
"""
# Author: Haoze Li <li4456@purdue.edu>
# License: MIT

import math
import random
import numpy as np
from PyXAB.algos.Algo import Algorithm
from PyXAB.partition.Node import P_node
import pdb


class SequOOL_node(P_node):
    def __init__(self, depth, index, parent, domain):
        super(SequOOL_node, self).__init__(depth, index, parent, domain)

        self.rewards = []
        self.mean_reward = 0
        self.opened = False

    def update_reward(self, reward):
        self.rewards.append(reward)

    def get_reward(self):
        return self.rewards[0]

    def open(self):
        self.opened = True

    def not_opened(self):
        return False if self.opened else True


class SequOOL(Algorithm):
    def __init__(self, n=1000, domain=None, partition=None):
        super(SequOOL, self).__init__()
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        self.partition = partition(domain=domain, node=SequOOL_node)
        self.iteration = 0

        self.h_max = math.floor(n / self.harmonic_series_sum(n))
        self.curr_depth = 0
        self.loc = 0
        self.open_loc = 0
        self.chosen = []

    @staticmethod
    def harmonic_series_sum(n):
        res = 0
        for i in range(1, n + 1):
            res += 1 / i
        return res

    def pull(self, t):
        node_list = self.partition.get_node_list()
        self.iteration = t

        if self.curr_depth <= self.h_max:
            if self.curr_depth == 0:
                node = node_list[0][0]
                if node.get_children() is None:
                    self.partition.make_children(node, newlayer=True)
                if self.loc < len(node.get_children()):
                    if self.loc == len(node.get_children()) - 1:
                        self.loc = 0
                        self.curr_depth += 1
                        self.budget = math.floor(self.h_max / self.curr_depth)
                        self.chosen.append(node.get_children()[-1])
                        self.curr_node = node.get_children()[-1]
                        return node.get_children()[-1].get_cpoint()
                    else:
                        self.loc += 1
                        self.chosen.append(node.get_children()[self.loc - 1])
                        self.curr_node = node.get_children()[self.loc - 1]
                        return node.get_children()[self.loc - 1].get_cpoint()
            else:
                max_value = -np.inf
                max_node = None
                num = 0
                for i in range(len(node_list[self.curr_depth])):
                    node = node_list[self.curr_depth][i]
                    if node.not_opened():
                        num += 1
                        if node.get_reward() >= max_value:
                            max_value = node.get_reward()
                            max_node = node

                if max_node.get_children() is None:
                    self.partition.make_children(max_node, newlayer=True)
                if self.loc < len(max_node.get_children()):
                    if self.loc == len(max_node.get_children()) - 1:
                        max_node.open()
                        self.loc = 0
                        self.budget -= 1
                        if self.budget == 0 or num == 1:
                            self.curr_depth += 1
                            self.budget = math.floor(self.h_max / self.curr_depth)
                        self.curr_node = max_node.get_children()[-1]
                        self.chosen.append(max_node.get_children()[-1])
                        return max_node.get_children()[-1].get_cpoint()
                    else:
                        self.loc += 1
                        self.curr_node = max_node.get_children()[self.loc - 1]
                        self.chosen.append(max_node.get_children()[self.loc - 1])
                        return max_node.get_children()[self.loc - 1].get_cpoint()
        else:
            self.curr_node = node_list[0][0]
            return node_list[0][0].get_cpoint()

    def receive_reward(self, t, reward):
        self.curr_node.update_reward(reward)

    def get_last_point(self):
        max_node = None
        max_value = -np.inf

        for node in self.chosen:
            if node.get_reward() >= max_value:
                max_node = node
                max_value = node.get_reward()

        return max_node.get_cpoint()


from PyXAB.synthetic_obj import *

# from PyXAB.algos.StroquOOL import StroquOOL
from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import compare_regret

# import math
# import numpy as np
# import pdb
T = 500
Target = Garland.Garland()
# Target = DoubleSine.DoubleSine()
domain = [[0, 1]]
partition = BinaryPartition
algo = SequOOL(n=T, domain=domain, partition=partition)

for t in range(1, T + 1):
    point = algo.pull(t)
    reward = Target.f(point)
    algo.receive_reward(t, reward)

last_point = algo.get_last_point()
print(algo.iteration, Target.fmax - Target.f(last_point), last_point)
