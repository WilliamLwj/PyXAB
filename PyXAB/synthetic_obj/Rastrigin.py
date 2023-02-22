# -*- coding: utf-8 -*-
"""Implmentation of Rastrigin
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT


import numpy as np
from PyXAB.synthetic_obj.Objective import Objective


class Rastrigin(Objective):
    """
    Rastrigin objective implementation, with the domain [-1, 1]^p
    """

    def __init__(self):
        """
        Initialization with fmax = 0
        """
        self.fmax = 0

    def f(self, x):
        """
        Evaluation of the chosen point in Rastrigin function

        Parameters
        ----------
        x: list
            one input point in the form of x = [x1, x2,...,xp]

        Returns
        -------
        y: float
            Evaluated value of the function at the particular point x = [x1, x2,...,xp], returns
            \sum_i^p  - 10 - (x[i] ** 2 - 10 * cos(2 * \pi * x[i]))
        """
        x = np.array(x)
        S = 0
        for i in range(x.size):
            S = S - 10 - (x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i]))

        return S


class Rastrigin_Normalized(Objective):
    """
    Normalized Rastrigin objective implementation, with the domain [-1, 1]^p
    """

    def __init__(self, k=20):
        """
        Initialization with fmax = 0, normalization constant k

        Parameters
        ----------
        k: float
            Normalization constant k
        """
        self.fmax = 0
        self.k = k  # Normalization constant

    def f(self, x):
        """
        Evaluation of the chosen point in normalized Rastrigin function

        Parameters
        ----------
        x: list
            one input point in the form of x = [x1, x2,...,xp]

        Returns
        -------
        y: float
            Evaluated value of the function at the particular point x = [x1, x2,...,xp], returns
            (1/pk)\sum_i^p  - 10 - (x[i] ** 2 - 10 * cos(2 * \pi * x[i]))
        """
        x = np.array(x)
        S = 0
        for i in range(x.size):
            S = S - 10 - (x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i]))

        S = S / (self.k * x.size)
        # Only devided by an upper bound in [-1, 1]^n

        return S
