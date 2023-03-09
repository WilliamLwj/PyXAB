# -*- coding: utf-8 -*-
"""Implmentation of Garland
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT


import numpy as np
from PyXAB.synthetic_obj.Objective import Objective


class Garland(Objective):
    """
    Garland objective implementation, with the domain [0, 1]
    """

    def __init__(self):
        """
        Initialization with fmax = 1
        """
        self.fmax = (
            1  # TODO: The actual maximum is not 1 but very close to 1, update this
        )

    def f(self, x):
        """
        Evaluation of the chosen point in Garland function

        Parameters
        ----------
        x: list
            one input point in the form of x = [x1]

        Returns
        -------
        y: float
            Evaluated value of the function at the particular point x = [x1], returns
            x * (1 - x) * (4 - sqrt(abs(sin(60 * x))))
        """
        if len(x) != 1:
            raise ValueError("The dimension of the point should be 1 in Garland")
        x = x[0]

        return x * (1 - x) * (4 - np.sqrt(np.abs(np.sin(60 * x))))


# Garland function perturbed by Gaussian noise


class Perturbed_Garland(Objective):
    """
    Perturbed Garland objective implementation, with the domain [0, 1]
    """

    def __init__(self):
        """
        Initialization with fmax = 1 +  perturbation
        """
        self.perturb = np.random.normal(0, 1)

        self.fmax = 1 + self.perturb

    def f(self, x):
        """
        Evaluation of the chosen point in perturbed Garland function

        Parameters
        ----------
        x: list
            one input point in the form of x = [x1]

        Returns
        -------
        y: float
            Evaluated value of the function at the particular point x = [x1], returns
            x * (1 - x) * (4 - sqrt(abs(sin(60 * x)))) + perturbation
        """
        if len(x) != 1:
            raise ValueError(
                "The dimension of the point should be 1 in Perturbed Garland"
            )

        x = x[0]

        return x * (1 - x) * (4 - np.sqrt(np.abs(np.sin(60 * x)))) + self.perturb
