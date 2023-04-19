# -*- coding: utf-8 -*-
"""Implementation of SequOOL (Bartlett, 2019)
"""
# Author: Haoze Li <li4456@purdue.edu>
# License: MIT

import math
import random
import numpy as np
from PyXAB.algos.Algo import Algorithm
from PyXAB.partition.Node import P_node
import pdb


class SequOOL_node(P_node):
    """
    Implementation of the SequOOL node
    """

    def __init__(self, depth, index, parent, domain):
        """
        Initialization of the SequOOL node
        Parameters
        ----------
        depth: int
            fepth of the node
        index: int
            index of the node
        parent: 
            parent node of the current node
        domain: list(list)
            domain that this node represents
        """
        super(SequOOL_node, self).__init__(depth, index, parent, domain)

        self.rewards = []
        self.mean_reward = 0
        self.opened = False

    def update_reward(self, reward):
        """
        The function to update the reward list of the node
        
        Parameters
        ----------
        reward: float
            the reward for evaluating the node
        
        Returns
        -------
        
        """
        self.rewards.append(reward)

    def get_reward(self):
        """
        The function to get the reward of the node

        Returns
        -------
        
        """
        return self.rewards[0]

    def open(self):
        """
        The function to open a node
        
        Returns
        -------
        
        """
        self.opened = True

    def not_opened(self):
        """
        The function to get the status of the node (opened or not)

        Returns
        -------
        
        """
        return False if self.opened else True


class SequOOL(Algorithm):
    """
    The implementation of the SequOOL algorithm (Barlett, 2019)
    """

    def __init__(self, n=1000, domain=None, partition=None):
        """
        The initialization of the SequOOL algorithm
        
        Parameters
        ----------
        n: int
            The totdal number of rounds (budget)
        domain: list(list)
            The domain of the objective to be optimized
        partition:
            The partition choice of the algorithm
        """
        super(SequOOL, self).__init__()
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        self.partition = partition(domain=domain, node=SequOOL_node)
        self.iteration = 0

        self.h_max = math.floor(n / self.harmonic_series_sum(n))
        self.curr_depth = 0
        self.loc = 0
        self.open_loc = 0
        self.chosen = []

    @staticmethod
    def harmonic_series_sum(n):
        """
        A static method for computing the summation of harmonic series
        
        Parameters
        ----------
        n: int
            The number of terms in the summation
        
        Returns
        -------
        res: float
            The sum of the series
        """
        res = 0
        for i in range(1, n + 1):
            res += 1 / i
        return res

    def pull(self, t):
        """
        The pull function of SequOOL that returns a point in every round
        
        Parameters
        ----------
        time: int
            time stamp parameter
        
        Returns
        -------
        point: list
            the point to be evaluated
        """
        node_list = self.partition.get_node_list()
        self.iteration = t

        if self.curr_depth <= self.h_max:
            if self.curr_depth == 0:
                node = node_list[0][0]
                if node.get_children() is None:
                    self.partition.make_children(node, newlayer=True)
                if self.loc < len(node.get_children()):
                    if self.loc == len(node.get_children()) - 1:
                        self.loc = 0
                        self.curr_depth += 1
                        self.budget = math.floor(self.h_max / self.curr_depth)
                        self.chosen.append(node.get_children()[-1])
                        self.curr_node = node.get_children()[-1]
                        return node.get_children()[-1].get_cpoint()
                    else:
                        self.loc += 1
                        self.chosen.append(node.get_children()[self.loc - 1])
                        self.curr_node = node.get_children()[self.loc - 1]
                        return node.get_children()[self.loc - 1].get_cpoint()
            else:
                max_value = -np.inf
                max_node = None
                num = 0
                for i in range(len(node_list[self.curr_depth])):
                    node = node_list[self.curr_depth][i]
                    if node.not_opened():
                        num += 1
                        if node.get_reward() >= max_value:
                            max_value = node.get_reward()
                            max_node = node

                if max_node.get_children() is None:
                    self.partition.make_children(max_node, newlayer=True)
                if self.loc < len(max_node.get_children()):
                    if self.loc == len(max_node.get_children()) - 1:
                        max_node.open()
                        self.loc = 0
                        self.budget -= 1
                        if self.budget == 0 or num == 1:
                            self.curr_depth += 1
                            self.budget = math.floor(self.h_max / self.curr_depth)
                        self.curr_node = max_node.get_children()[-1]
                        self.chosen.append(max_node.get_children()[-1])
                        return max_node.get_children()[-1].get_cpoint()
                    else:
                        self.loc += 1
                        self.curr_node = max_node.get_children()[self.loc - 1]
                        self.chosen.append(max_node.get_children()[self.loc - 1])
                        return max_node.get_children()[self.loc - 1].get_cpoint()
        else:
            self.curr_node = node_list[0][0]
            return node_list[0][0].get_cpoint()

    def receive_reward(self, t, reward):
        """
        The receive_reward function of SequOOL to obtain the reward and update Statistics
        
        Parameters
        ----------
        t: int
            The time stamp parameter
        reward: float
            The reward of the evaluation
        
        Returns
        -------
        
        """
        self.curr_node.update_reward(reward)

    def get_last_point(self):
        """
        The function to get the last point in SequOOL
        
        Returns
        -------
        point: list
            The output of the SequOOL algorithm at last
        """
        max_node = None
        max_value = -np.inf

        for node in self.chosen:
            if node.get_reward() >= max_value:
                max_node = node
                max_value = node.get_reward()

        return max_node.get_cpoint()
