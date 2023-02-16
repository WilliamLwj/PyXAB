# -*- coding: utf-8 -*-
"""Implmentation of the counter example
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT


import numpy as np
from PyXAB.synthetic_obj.Objective import Objective


class Cexample(Objective):
    def __init__(self):
        self.fmax = 1

    def f(self, x):
        x = x[0]
        return 1 + 1 / np.log(x)
