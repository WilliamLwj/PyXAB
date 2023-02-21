# -*- coding: utf-8 -*-
"""Base algorithm for X-Armed bandit problems
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT

import math
import numpy as np
import pdb
from abc import ABC, abstractmethod


class Algorithm(ABC):
    """
    Abstract class for X-armed bandit algorithms.
    """

    @abstractmethod
    def __init__(self):
        """
        Initialization for the algorithm
        """
        pass

    @abstractmethod
    def pull(self, time):
        """
        Every algorithm needs a function to pull a node.

        Parameters
        ----------
        time: int
            The time step of the online process.

        Returns
        -------
        chosen_point: list
            The point chosen by the algorithm
        """
        pass

    @abstractmethod
    def receive_reward(self, time, reward):
        """
        Every algorithm needs a function to receive the reward.

        Parameters
        ----------
        time: int
            The time step of the online process.

        reward: float
            The (Stochastic) reward of the pulled point

        Returns
        -------
        """
        pass
