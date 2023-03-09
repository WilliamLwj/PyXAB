# -*- coding: utf-8 -*-
"""Implmentation of DoubleSine
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT

import math
import numpy as np
from PyXAB.synthetic_obj.Objective import Objective


def mysin2(x):
    return (np.sin(x * 2 * np.pi) + 1) / 2.0


class DoubleSine(Objective):
    """
    DoubleSine objective implementation, with the domain [0, 1]
    """

    def __init__(self, rho1=0.3, rho2=0.8, tmax=0.5):
        """
        Initialization with fmax = 0

        Parameters
        ----------
        rho1: float
            The parameter rho1 between 0 and 1 to compute ep1
        rho2: float
            The parameter rho2 between 0 and 1 to compute ep2
        tmax: float
            The parameter tmax between 0 and 1 to truncate x
        """
        if rho1 <= 0 or rho1 > 1:
            raise ValueError("The parameter rho_1 should satisfy 0 < rho1 <= 1")
        if rho2 <= 0 or rho2 > 1:
            raise ValueError("The parameter rho_2 should satisfy 0 < rho2 <= 1")
        if tmax < 0 or tmax > 1:
            raise ValueError("The parameter tmax should satisfy 0 <= tmax <= 1")

        self.ep1 = -math.log(rho1, 2)
        self.ep2 = -math.log(rho2, 2)
        self.tmax = tmax
        self.fmax = 0.0

    def f(self, x):
        """
        Evaluation of the chosen point in DoubleSine function

        Parameters
        ----------
        x: list
            one input point in the form of x = [x1]

        Returns
        -------
        y: float
            Evaluated value of the function at the particular point x = [x1], returns
            mysin2(log(u, 2) / 2.0) * envelope_width - pow(u, self.ep2)
        """
        if len(x) != 1:
            raise ValueError("The dimension of the point should be 1 in DoubleSine")
        x = x[0]
        u = 2 * np.fabs(x - self.tmax)
        if u == 0:
            return 0.0
        else:
            envelope_width = math.pow(u, self.ep2) - math.pow(u, self.ep1)
            return mysin2(math.log(u, 2) / 2.0) * envelope_width - math.pow(u, self.ep2)


# DoubleSine function perturbed by Gaussian noise


class Perturbed_DoubleSine(Objective):
    """
    Perturbed DoubleSine objective implementation, with the domain [0, 1]
    """

    def __init__(self, rho1=0.3, rho2=0.8, tmax=0.5):
        """
        Initialization with fmax = 0 + perturbation

        Parameters
        ----------
        rho1: float
            The parameter rho1 between 0 and 1 to compute ep1
        rho2: float
            The parameter rho2 between 0 and 1 to compute ep2
        tmax: float
            The parameter tmax between 0 and 1 to truncate x

        """
        if rho1 <= 0 or rho1 > 1:
            raise ValueError("The parameter rho_1 should satisfy 0 < rho1 <= 1")
        if rho2 <= 0 or rho2 > 1:
            raise ValueError("The parameter rho_2 should satisfy 0 < rho2 <= 1")
        if tmax < 0 or tmax > 1:
            raise ValueError("The parameter tmax should satisfy 0 <= tmax <= 1")

        self.ep1 = -math.log(rho1, 2)
        self.ep2 = -math.log(rho2, 2)
        self.tmax = tmax
        self.perturb = np.random.normal(0, 1)

        self.fmax = 0.0 + self.perturb

    def f(self, x):
        """
        Evaluation of the chosen point in perturbed DoubleSine function

        Parameters
        ----------
        x: list
            one input point in the form of x = [x1]

        Returns
        -------
        y: float
            Evaluated value of the function at the particular point x = [x1], returns
            mysin2(log(u, 2) / 2.0) * envelope_width - pow(u, self.ep2) + perturbation
        """
        if len(x) != 1:
            raise ValueError(
                "The dimension of the point should be 1 in Perturbed DoubleSine"
            )
        x = x[0]
        u = 2 * np.fabs(x - self.tmax)
        if u == 0:
            return 0.0 + self.perturb
        else:
            envelope_width = math.pow(u, self.ep2) - math.pow(u, self.ep1)
            return (
                mysin2(math.log(u, 2) / 2.0) * envelope_width
                - math.pow(u, self.ep2)
                + self.perturb
            )
