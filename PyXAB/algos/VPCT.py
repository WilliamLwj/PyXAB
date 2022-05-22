# -*- coding: utf-8 -*-
"""Implementation of VPCT (Li et al. 2021)
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT

import numpy as np
from PyXAB.algos.Algo import Algorithm
from PyXAB.algos.VHCT import VHCT
from PyXAB.algos.GPO import GPO

class VPCT(Algorithm):

    def __init__(self, numax=1, rhomax=0.9, rounds=1000, domain=None, partition=None):
        super(VPCT, self).__init__()
        self.algorithm = GPO(numax=numax, rhomax=rhomax, rounds=rounds, domain=domain, partition=partition, algo=VHCT)
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")


    def pull(self, time):

        return self.algorithm.pull(time)

    def receive_reward(self, time, reward):

        return self.algorithm.receive_reward(time, reward)
