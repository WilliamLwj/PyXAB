# -*- coding: utf-8 -*-
"""Implmentation of Himmelblau
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT


from PyXAB.synthetic_obj.Objective import Objective


class Himmelblau(Objective):
    def __init__(self):
        self.fmax = 0

    def f(self, x):
        x1 = x[0]
        x2 = x[1]
        return -((x1**2 + x2 - 11) ** 2) - (x1 + x2**2 - 7) ** 2


class Himmelblau_Normalized(Objective):
    def __init__(self):
        self.fmax = 0

    def f(self, x):
        x1 = x[0]
        x2 = x[1]

        S = -((x1**2 + x2 - 11) ** 2) - (x1 + x2**2 - 7) ** 2
        # Only devided by an upper bound in [-5, 5]^2

        return S / 890
