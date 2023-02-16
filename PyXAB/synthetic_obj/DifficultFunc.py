# -*- coding: utf-8 -*-
"""Implmentation of Difficult Function
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT


import numpy as np
from PyXAB.synthetic_obj.Objective import Objective


def threshold(x):
    if x - np.floor(x) < 0.5:
        x = 0
    else:
        x = 1

    return x


class DifficultFunc(Objective):
    def __init__(self):
        self.fmax = 0.0

    def f(self, x):
        x = x[0]
        y = np.abs(x - 0.5)
        if y == 0:
            return 0
        else:
            return threshold(np.log(y)) * (np.sqrt(y) - y**2) - np.sqrt(y)

    def fmax(self):
        return self.fmax
