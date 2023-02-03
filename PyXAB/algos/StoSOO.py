# -*- coding: utf-8 -*-
"""Implementation of StoSOO (Munos, 2013)
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT


import math
import numpy as np
from PyXAB.algos.Algo import Algorithm
from PyXAB.partition.Node import P_node
import pdb


class StoSOO_node(P_node):

    def __init__(self, depth, index, parent, domain):
        super(StoSOO_node, self).__init__(depth, index, parent, domain)
        self.depth = depth
        self.index = index
        self.parent = parent
        self.children = None
        self.domain = domain

        point = []
        for x in self.domain:

            point.append((x[0] + x[1]) / 2)# TODO: Different Domains Other Than Continuous Domains

        self.c_point = point


class StoSOO(Algorithm):

    def __init__(self, n=100, k=5, h_max=100, delta=0.01, domain=None, partition=None):
        super(StoSOO, self).__init__()
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        self.partition = partition(domain=domain)

        self.iteration = 0
        self.n = n       # budget
        self.k = k      # max number of pulls per node
        self.delta = delta
        self.h_max = h_max      # max depth


    def pull(self, time):
        if time > self.n:
            return self.get_last_point()

        self.b_max = - np.inf
        self.iteration = time
        for h in range(min(self.partition.get_depth(), self.h_max)):
            if self.iteration <= self.n:
                return 0


    def receive_reward(self, time, reward):

        pass

    def get_last_point(self):

        pass