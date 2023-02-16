# -*- coding: utf-8 -*-
"""Implmentation of Garland
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT


import numpy as np
from PyXAB.synthetic_obj.Objective import Objective


class Garland(Objective):
    def __init__(self):
        self.fmax = 1

    def f(self, x):
        x = x[0]

        return x * (1 - x) * (4 - np.sqrt(np.abs(np.sin(60 * x))))


# Garland function perturbed by Gaussian noise


class Perturbed_Garland(Objective):
    def __init__(self):
        self.perturb = np.random.normal(0, 1)

        self.fmax = 1 + self.perturb

    def f(self, x):
        x = x[0]

        return x * (1 - x) * (4 - np.sqrt(np.abs(np.sin(60 * x)))) + self.perturb
