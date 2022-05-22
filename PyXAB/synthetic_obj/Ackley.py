# -*- coding: utf-8 -*-
"""Implmentation of Ackley
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT

import numpy as np
from PyXAB.synthetic_obj.Objective import Objective

class Ackley(Objective):
    def __init__(self):

        self.fmax = 0

    def f(self, x):

        x1 = x[0]
        x2 = x[1]
        return 20 * np.exp(-0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2)))\
               + np.exp(0.5 * (np.cos(2*np.pi * x1) + np.cos(2*np.pi * x2))) - np.e - 20



class Ackley_Normalized(Objective):
    def __init__(self):

        self.fmax = 0

    def f(self, x):

        x1 = x[0]
        x2 = x[1]
        S = 20 * np.exp(-0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2)))\
               + np.exp(0.5 * (np.cos(2*np.pi * x1) + np.cos(2*np.pi * x2))) - np.e - 20


        # Only devided by an upper bound in [-1, 1]^2
        return S / np.abs(20 * np.exp(-0.2 * np.sqrt(0.5)) + np.exp(-1) - 20 - np.exp(1))