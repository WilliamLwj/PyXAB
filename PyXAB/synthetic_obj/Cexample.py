# -*- coding: utf-8 -*-
"""Implmentation of the counter example
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT


import numpy as np
from PyXAB.synthetic_obj.Objective import Objective


class Cexample(Objective):
    """
    An example of objective failing exponential smoothness, with the domain [0, 1/e]
    """

    def __init__(self):
        """
        Initialization of fmax = 1
        """
        self.fmax = 1

    def f(self, x):
        """
        Evaluation of the chosen point in the objective function

        Parameters
        ----------
        x: list
            one input point in the form of x = [x1]

        Returns
        -------
        y: float
            Evaluated value of the function at the particular point x = [x1], returns
            1 + 1 / log(x)
        """
        if len(x) != 1:
            raise ValueError("The dimension of the point should be 1 in Cexample")
        x = x[0]
        if x == 0:
            return 1
        else:
            return 1 + 1 / np.log(x)
