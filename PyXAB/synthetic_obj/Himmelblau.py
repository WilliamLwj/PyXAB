# -*- coding: utf-8 -*-
"""Implmentation of Himmelblau
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT


from PyXAB.synthetic_obj.Objective import Objective


class Himmelblau(Objective):
    """
    Himmelblau objective implementation, with the domain [-5, 5]^2
    """

    def __init__(self):
        """
        Initialization with fmax = 0
        """
        self.fmax = 0

    def f(self, x):
        """
        Evaluation of the chosen point in Himmelblau function

        Parameters
        ----------
        x: list
            one input point in the form of x = [x1, x2]

        Returns
        -------
        y: float
            Evaluated value of the function at the particular point x = [x1, x2], returns
           -((x1**2 + x2 - 11) ** 2) - (x1 + x2**2 - 7) ** 2
        """
        if len(x) != 2:
            raise ValueError("The dimension of the point should be 2 in Himmelblau")
        x1 = x[0]
        x2 = x[1]
        return -((x1 ** 2 + x2 - 11) ** 2) - (x1 + x2 ** 2 - 7) ** 2


class Himmelblau_Normalized(Objective):
    """
    Normalized Himmelblau objective implementation, with the domain [-5, 5]^2, normalized so that the response is
    between [0, 1]
    """

    def __init__(self):
        """
        Initialization with fmax = 0
        """
        self.fmax = 0

    def f(self, x):
        """
        Evaluation of the chosen point in the normalized Himmelblau function

        Parameters
        ----------
        x: list
            one input point in the form of x = [x1, x2]

        Returns
        -------
        y: float
            Evaluated value of the function at the particular point x = [x1, x2], returns
            [-((x1**2 + x2 - 11) ** 2) - (x1 + x2**2 - 7) ** 2]/ 890
        """
        if len(x) != 2:
            raise ValueError(
                "The dimension of the point should be 2 in Himmelblau_Normalized"
            )
        x1 = x[0]
        x2 = x[1]

        S = -((x1 ** 2 + x2 - 11) ** 2) - (x1 + x2 ** 2 - 7) ** 2
        # Only devided by an upper bound in [-5, 5]^2

        return S / 890
