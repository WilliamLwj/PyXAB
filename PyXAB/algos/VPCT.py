# -*- coding: utf-8 -*-
"""Implementation of VPCT
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT

import numpy as np
from PyXAB.algos.Algo import Algorithm
from PyXAB.algos.VHCT import VHCT
from PyXAB.algos.GPO import GPO
from PyXAB.partition.BinaryPartition import BinaryPartition

class VPCT(Algorithm):
    """
    Implementation of Variance-reduced Parallel Confidence Tree  algorithm (VHCT + GPO)
    """

    def __init__(self, numax=1, rhomax=0.9, rounds=1000, domain=None, partition=BinaryPartition):
        """
        Initialization of the VPCT algorithm

        Parameters
        ----------
        numax: float
            parameter nu_max in the algorithm
        rhomax: float
            parameter rho_max in the algorithm, the maximum rho used
        rounds: int
            the number of rounds/budget
        domain: list(list)
            the domain of the objective function
        partition:
            the partition used in the optimization process
        """
        super(VPCT, self).__init__()

        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")

        self.algorithm = GPO(
            numax=numax,
            rhomax=rhomax,
            rounds=rounds,
            domain=domain,
            partition=partition,
            algo=VHCT,
        )

    def pull(self, time):
        """
        The pull function of VPCT that returns a point to be evaluated

        Parameters
        ----------
        time: int
            The time step of the online process.

        Returns
        -------
        point: list
            The point chosen by the VPCT algorithm

        """

        return self.algorithm.pull(time)

    def receive_reward(self, time, reward):
        """
        The receive_reward function of VPCT to receive the reward for the chosen point

        Parameters
        ----------
        time: int
            The time step of the online process.

        reward: float
            The (Stochastic) reward of the pulled point

        Returns
        -------

        """
        self.algorithm.receive_reward(time, reward)

    def get_last_point(self):
        """
        The function to get the last point of VPCT.

        Returns
        -------

        """
        return self.algorithm.get_last_point()
