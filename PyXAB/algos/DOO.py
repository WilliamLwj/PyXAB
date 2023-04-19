# -*- coding: utf-8 -*-
"""Implementation of DOO (Munos, 2011)
"""
# Author: Haoze Li <li4456@purdue.edu>
# License: MIT


import math
import numpy as np
from PyXAB.algos.Algo import Algorithm
from PyXAB.partition.Node import P_node
import pdb


class DOO_node(P_node):
    """
    Implementation of the node in the DOO algorithm
    """

    def __init__(self, depth, index, parent, domain):
        """
        Initialization of the DOO node
        
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
        super(DOO_node, self).__init__(depth, index, parent, domain)

        self.b_value = np.inf
        self.reward = 0
        self.visited = False

    def update_reward(self, reward):
        """
        The function to update the reward of the node
        
        Parameters
        ----------
        reward: float
            the reward for evaluating the node
        
        Returns
        -------
        
        """
        self.reward = reward

    def compute_b_value(self, delta):
        """
        The function to compute the b_{h,i} value of the node
        
        Parameters
        ----------
        delta: float
            The delta value in the b_{h,i} term, which depends on the depth of the node (Munos, 2011)

        Returns
        -------
        
        """
        self.b_value = self.reward + delta

    def get_b_value(self):
        """
        The function to get the b_{h,i} value of the node
        
        Returns
        -------
        
        """
        return self.b_value

    def visit(self):
        """
        The function to visit the node
        
        Returns
        -------
        
        """
        self.visited = True

    def get_reward(self):
        """
        The function to get the reward of the node
        
        Returns
        -------

        """
        return self.reward


class DOO(Algorithm):
    """
    The implementation of the DOO algorithm (Munos, 2011)
    """

    def __init__(self, n=100, delta=None, domain=None, partition=None):
        """
        The initialization of the DOO algorithm

        Parameters
        ----------
        n: int
            The total number of rounds (budget)
        delta: function
            The function to compute the delta value for each depth
        domain: list(list)
            The domain of the objective to be optimized
        partition: 
            The partition choice of the algorithm
        """
        super(DOO, self).__init__()
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        self.partition = partition(domain=domain, node=DOO_node)

        if delta is None:
            # a function of h that returns delta
            self.delta = self.delta_init
        self.iteration = 0
        self.n = n

    def pull(self, time):
        """
        The pull function of DOO that returns a point in every round

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
        node_list = self.partition.get_node_list()

        max_node = None
        max_value = -np.inf

        h = 0
        while h <= self.partition.get_depth():
            delta = self.delta(h)
            for node in node_list[h]:
                if node.get_children() is None:
                    if node.visited:
                        node.compute_b_value(delta)
                        if node.get_b_value() >= max_value:
                            max_value = node.get_b_value()
                            max_node = node
                    else:
                        node.visit()
                        self.curr_node = node
                        return node.get_cpoint()
            h += 1
            if h > self.partition.get_depth():
                self.partition.make_children(max_node, newlayer=True)
                h = 0

    def receive_reward(self, time, reward):
        """
        The receive_reward function of DOO to obtain the reward and update Statistics

        Parameters
        ----------
        time: int 
            The time stamp parameter
        reward: float
            The reward of the evaluation
        
        Returns
        -------
        
        """
        self.curr_node.update_reward(reward)

    def get_last_point(self):
        """
        The function to get the last point in DOO

        Returns
        -------
        point: list
            The output of the DOO algorithm at last
        """
        max_value = -np.inf
        max_node = None

        for i in self.partition.get_node_list():
            for node in i:
                reward = node.get_reward()
                if reward >= max_value:
                    max_value = reward
                    max_node = node

        return max_node.get_cpoint()

    def delta_init(self, h):  # delta that satisfies Assumption 3
        """
        The default delta function used in the algorithm (Munos, 2011)

        Parameters
        ----------
        h: int
            The depth parameter
        
        Returns
        -------
        max_value: float
            The delta value in that depth
        """
        node_list = self.partition.get_node_list()

        max_value = -np.inf

        for node in node_list[h]:
            domain = node.get_domain()
            point = node.get_cpoint()[0]
            if (
                max((domain[0][0] - point) ** 2, (domain[0][1] - point) ** 2)
                >= max_value
            ):
                max_value = max(
                    (domain[0][0] - point) ** 2, (domain[0][1] - point) ** 2
                )

        return max_value
