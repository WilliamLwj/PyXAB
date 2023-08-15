# -*- coding: utf-8 -*-
"""Implementation of SOO (Munos, 2011)
"""
# Author: Haoze Li <li4456@purdue.edu>
# License: MIT


import math
import numpy as np
from PyXAB.algos.Algo import Algorithm
from PyXAB.partition.Node import P_node
from PyXAB.partition.BinaryPartition import BinaryPartition
import pdb


class SOO_node(P_node):
    """
    Implementation of the node in the SOO algorithm
    """

    def __init__(self, depth, index, parent, domain):
        """
        Initialization of the SOO node
        
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
        super(SOO_node, self).__init__(depth, index, parent, domain)

        self.visited = False
        self.reward = -np.inf

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

    def get_reward(self):
        """
        The function to get the reward of the node
        
        Returns
        -------

        """
        return self.reward

    def visit(self):
        """
        The function to visit the node
        
        Returns
        -------
        
        """

        self.visited = True


class SOO(Algorithm):
    """
    The implementation of the SOO algorithm (Munos, 2011)
    """

    def __init__(self, n=100, h_max=100, domain=None, partition=BinaryPartition):
        """
        The initialization of the SOO algorithm
        
        Parameters
        ----------
        n: int
            The total number of rounds (budget)
        h_max: int
            The largest searching depth
        domain: list(list)
            The domain of the objective to be optimized
        partition: 
            The partition choice of the algorithm
        """
        super(SOO, self).__init__()
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        self.partition = partition(domain=domain, node=SOO_node)

        self.iteration = 0
        self.n = n
        self.h_max = h_max

        self.curr_node = None

    def pull(self, time):
        """
        The pull function of SOO that returns a point in every round

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

        while True:
            h = 0
            v_max = -np.inf
            while h <= min(self.partition.get_depth(), self.h_max):
                max_value = -np.inf
                max_node = None
                for node in node_list[h]:  # for all node in the layer
                    if (
                        node.get_children() is None
                    ):  # if the node is not evaluated, evaluate it
                        if not node.visited:
                            node.visit()
                            self.curr_node = node
                            return node.get_cpoint()
                        if (
                            node.get_reward() >= max_value
                        ):  # find the leaf node with maximal reward
                            max_value = node.get_reward()
                            max_node = node
                if max_value >= v_max:
                    if max_node is not None:  # Found a leaf node
                        self.partition.make_children(
                            max_node, newlayer=(h >= self.partition.get_depth())
                        )
                        v_max = max_value
                h += 1

    def receive_reward(self, time, reward):
        """
        The receive_reward function of SOO to obtain the reward and update Statistics (for current node)

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
        The function to get the last point in SOO

        Returns
        -------
        point: list
            The output of the SOO algorithm at last
        """

        max_value = -np.inf
        max_node = None
        node_list = self.partition.get_node_list()
        for h in range(len(node_list)):
            for node in node_list[h]:
                if node.get_reward() >= max_value:
                    max_value = node.get_reward()
                    max_node = node
        return max_node.get_cpoint()
