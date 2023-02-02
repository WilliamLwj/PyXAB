# -*- coding: utf-8 -*-
"""Implementation of StoSOO (Munos, 2013)
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT


import math
import numpy as np
from PyXAB.algos.Algo import Algorithm
import pdb

class StoSOO(Algorithm):

    def __init__(self, n=100, k=5, h_max=100, delta=0.01, domain=None, partition=None):
        super(StoSOO, self).__init__()
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        self.partition = partition(domain=domain)

        self.iteration = 1
        self.n = n       # budget
        self.k = k      # max number of pulls per node
        self.delta = delta
        self.h_max = h_max      # max depth


    def pull(self, time):


        pass
        

    def receive_reward(self, time, reward):

        pass

    def get_last_point(self):

        pass