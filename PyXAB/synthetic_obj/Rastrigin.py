# -*- coding: utf-8 -*-
"""Implmentation of Rastrigin
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT


import numpy as np
from PyXAB.synthetic_obj.Objective import Objective


class Rastrigin(Objective):
    def __init__(self):
        self.fmax = 0

    def f(self, x):
        x = np.array(x)
        S = 0
        for i in range(x.size):
            S = S - 10 - (x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i]))

        return S


class Rastrigin_Normalized(Objective):
    def __init__(self, k=20):
        self.fmax = 0
        self.k = k  # Normalization constant

    def f(self, x):
        x = np.array(x)
        S = 0
        for i in range(x.size):
            S = S - 10 - (x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i]))

        S = S / (self.k * x.size)
        # Only devided by an upper bound in [-1, 1]^n

        return S
