# -*- coding: utf-8 -*-
"""Implementation of StoSOO (Valko et al., 2013)
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT


import math
import numpy as np
from PyXAB.algos.Algo import Algorithm
from PyXAB.partition.Node import P_node
import pdb


class StoSOO_node(P_node):
    """
    Implementation of the node in the StoSOO algorithm
    """

    def __init__(self, depth, index, parent, domain):
        """
        Initialization of the StoSOO node

        Parameters
        ----------
        depth: int
            depth of the node
        index: int
            index of the node
        parent:
            parent node of the current node
        domain: list(list)
            domain that this node represents
        """
        super(StoSOO_node, self).__init__(depth, index, parent, domain)

        self.b_value = np.inf
        self.visited_times = 0
        self.rewards = []
        self.mean_reward = 0

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
        self.visited_times += 1
        self.rewards.append(reward)
        self.mean_reward = np.sum(np.array(self.rewards)) / self.visited_times

    def compute_b_value(self, n, k, delta):
        """
        The function to compute the b_{h,i} value of the node

        Parameters
        ----------
        n: int
            The total number of rounds (budget)
        k: int
            The maximum number of pulls per node
        delta: float
            The confidence parameter

        Returns
        -------

        """
        if self.visited_times == 0:
            self.b_value = np.inf
        else:
            self.mean_reward = np.sum(np.array(self.rewards)) / self.visited_times
            self.b_value = self.mean_reward + np.sqrt(
                np.log(n * k / delta) / (2 * self.visited_times)
            )

    def get_visited_times(self):
        """
        The function to get the number of visited times of the node

        Returns
        -------

        """
        return self.visited_times

    def get_b_value(self):
        """
        The function to get the b_{h,i} value of the node

        Returns
        -------

        """
        return self.b_value

    def get_mean_reward(self):
        """
        The function to get the mean reward of the node

        Returns
        -------

        """
        return self.mean_reward


class StoSOO(Algorithm):
    """
    The implementation of the StoSOO algorithm (Valko et al., 2013)
    """

    def __init__(
        self, n=100, k=None, h_max=100, delta=None, domain=None, partition=None
    ):
        """
        The initialization of the StoSOO algorithm

        Parameters
        ----------
        n: int
            The total number of rounds (budget)
        k: int
            The maximum number of pulls per node
        h_max: int
            The maximum depth limit
        delta: float
            The confidence parameter delta
        domain: list(list)
            The domain of the objective to be optimized
        partition:
            The partition choice of the algorithm
        """
        super(StoSOO, self).__init__()
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        self.partition = partition(domain=domain, node=StoSOO_node)

        self.iteration = 0
        self.n = n  # budget
        if k is None:  # max number of pulls per node
            self.k = np.ceil(self.n / (np.log(n) ** 3))
        else:
            self.k = k

        if delta is None:
            self.delta = 1 / np.sqrt(self.n)
        else:
            self.delta = delta
        self.h_max = h_max  # max depth

    def pull(self, time):
        """
        The pull function of StoSOO that returns a point in every round

        Parameters
        ----------
        time: int
            time stamp parameter

        Returns
        -------
        point: list
            the point to be evaluated
        """
        self.iteration = time
        self.b_max = -np.inf
        node_list = self.partition.get_node_list()

        h = 0
        while h <= (min(self.partition.get_depth() + 1, self.h_max)):
            if self.iteration <= self.n:
                max_b_node_ind = None
                for j in range(len(node_list[h])):  # for every node
                    node = node_list[h][j]
                    if node.get_children() is None:  # if it is a leaf
                        node.compute_b_value(n=self.n, k=self.k, delta=self.delta)

                        # Locate the leaf node with the max b
                        if max_b_node_ind is None:
                            max_b_node_ind = j
                        else:
                            if (
                                node_list[h][max_b_node_ind].get_b_value()
                                <= node_list[h][j].get_b_value()
                            ):
                                max_b_node_ind = j

                if max_b_node_ind is not None:  # If there exists a leaf node in layer h
                    if node_list[h][max_b_node_ind].get_b_value() >= self.b_max:
                        if node_list[h][max_b_node_ind].get_visited_times() < self.k:
                            self.max_b_node_ind = max_b_node_ind
                            self.max_b_node_h = h

                            return node_list[h][
                                max_b_node_ind
                            ].get_cpoint()  # Return a point
                        else:
                            # If there is a new layer, just add the children, otherwise create a new layer
                            self.partition.make_children(
                                node_list[h][max_b_node_ind],
                                newlayer=(h >= self.partition.get_depth()),
                            )
                            self.b_max = node_list[h][max_b_node_ind].get_b_value()

                h += 1  # increase the search depth

    def receive_reward(self, time, reward):
        """
        The receive_reward function of StoSOO to obtain the reward and update the Statistics

        Parameters
        ----------
        time: int
            The time stamp parameter
        reward: float
            the reward of the evaluation

        Returns
        -------

        """
        node_list = self.partition.get_node_list()
        node_list[self.max_b_node_h][self.max_b_node_ind].update_reward(reward)

    def get_last_point(self):
        """
        The function to get the last point in StoSOO

        Returns
        -------
        point: list
            The output of the StoSOO algorithm at last
        """
        max_depth = self.partition.get_depth()
        max_mu = -np.inf
        max_x = None
        for node in self.partition.get_node_list()[max_depth]:
            mean_reward = node.get_mean_reward()
            if mean_reward >= max_mu:
                max_mu = mean_reward
                max_x = node.get_cpoint()

        return max_x
