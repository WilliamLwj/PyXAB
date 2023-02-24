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
    """
    DifficultFunc objective implementation, with the domain [0, 1]
    """

    def __init__(self):
        """
        Initialization with fmax = 1
        """
        self.fmax = 0.0

    def f(self, x):
        """
        Evaluation of the chosen point in DifficultFunc function

        Parameters
        ----------
        x: list
            one input point in the form of x = [x1]

        Returns
        -------
        y: float
            Evaluated value of the function at the particular point x = [x1], returns
            threshold(log(y)) * (sqrt(y) - y**2) - sqrt(y)
        """
        if len(x) != 1:
            raise ValueError("The dimension of the point should be 1 in DifficultFunc")
        x = x[0]
        y = np.abs(x - 0.5)
        if y == 0:
            return 0
        else:
            return threshold(np.log(y)) * (np.sqrt(y) - y ** 2) - np.sqrt(y)
