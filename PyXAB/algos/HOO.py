# -*- coding: utf-8 -*-
"""Implementation of T-HOO, the truncated version of the HOO algorithm. (Bubeck et al, 2011)
    The original HOO algorithm suffers from large space complexity.
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT

import math
import numpy as np
from PyXAB.algos.Algo import Algorithm
import pdb


class T_HOO(Algorithm):
    def __init__(self, nu=1, rho=0.5, rounds=1000, domain=None, partition=None):
        super(T_HOO, self).__init__()
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        self.partition = partition(domain=domain)

        self.iteration = 0
        self.nu = nu
        self.rho = rho
        self.rounds = rounds

        # List of values that are important

        self.Bvalues = [[np.inf]]
        self.Uvalues = [[np.inf]]
        self.Rewards = [[0]]
        self.visitedTimes = [[0]]
        self.visited = [[True]]
        self.expand(self.partition.get_root())

    def optTraverse(self):
        """
        Traverse the exploration tree to find the best path and the best node to pull at this moment.

        Returns
        -------
        curr_node: Node
            The last node selected by the algorithm
        path: List of Node
            The best path to traverse the partition selected by the algorithm
        """

        curr_node = self.partition.get_root()
        path = [curr_node]

        while curr_node.get_children() is not None:
            children = curr_node.get_children()
            maxchild = None
            maxindex = children[
                0
            ].get_index()  # temporarily set the maxindex to be the first child
            for child in children:
                c_depth = child.get_depth()
                c_index = child.get_index()

                # If the child is never visited or prepared to be visited, denote maxchild = None and break
                if not self.visited[c_depth][c_index - 1]:
                    maxchild = None
                    break
                elif (
                    self.Bvalues[c_depth][c_index - 1]
                    >= self.Bvalues[c_depth][maxindex - 1]
                ):
                    maxchild = child
                    maxindex = c_index

            # If we find that the child is never visited, stop going deeper
            if maxchild is None:
                break
            else:
                curr_node = maxchild
                path.append(maxchild)

        return curr_node, path

    def updateRewardTree(self, path, reward):
        for node in path:
            depth = node.get_depth()
            index = node.get_index()

            # Update the visited times and the average reward of the pulled node

            self.visitedTimes[depth][index - 1] += 1
            self.Rewards[depth][index - 1] = (
                (self.visitedTimes[depth][index - 1] - 1)
                / self.visitedTimes[depth][index - 1]
                * self.Rewards[depth][index - 1]
            ) + (reward / self.visitedTimes[depth][index - 1])

        self.iteration += 1

    def updateUvalueTree(self):
        node_list = self.partition.get_node_list()
        for layer in node_list:
            for node in layer:
                depth = node.get_depth()
                index = node.get_index()

                if self.visitedTimes[depth][index - 1] == 0:
                    continue
                else:
                    UCB = math.sqrt(
                        2 * math.log(self.rounds) / self.visitedTimes[depth][index - 1]
                    )
                    self.Uvalues[depth][index - 1] = (
                        self.Rewards[depth][index - 1]
                        + UCB
                        + self.nu * (self.rho**depth)
                    )

    def updateBackwardTree(self):
        nodes = self.partition.get_node_list()

        for i in range(1, self.partition.get_depth() + 1):
            layer = nodes[-i]
            for node in layer:
                depth = node.get_depth()
                index = node.get_index()

                # If no children or if children not visitied, use its own U value
                children = node.get_children()
                if children is None:
                    self.Bvalues[depth][index - 1] = self.Uvalues[depth][index - 1]
                else:
                    c_depth = children[0].depth
                    c_index = children[0].index
                    if not self.visited[c_depth][c_index]:
                        self.Bvalues[depth][index - 1] = self.Uvalues[depth][index - 1]
                    else:
                        tempB = 0
                        for child in node.get_children():
                            c_depth = child.get_depth()
                            c_index = child.get_index()
                            tempB = np.maximum(
                                tempB, self.Bvalues[c_depth][c_index - 1]
                            )

                        self.Bvalues[depth][index - 1] = np.minimum(
                            self.Uvalues[depth][index - 1], tempB
                        )

    def expand(self, parent):
        if parent.get_depth() > self.partition.get_depth():
            raise ValueError("parent depth larger than partition depth")
        elif parent.get_depth() == self.partition.get_depth():
            self.partition.deepen()
            num_nodes = len(self.partition.get_node_list()[-1])
            self.Uvalues.append([np.inf] * num_nodes)
            self.Bvalues.append([np.inf] * num_nodes)
            self.visited.append([False] * num_nodes)
            self.visitedTimes.append([0] * num_nodes)
            self.Rewards.append([0] * num_nodes)

        children = parent.get_children()
        if children is None:
            raise ValueError("No Children")
        else:
            for child in children:
                c_depth = child.get_depth()
                c_index = child.get_index()
                self.visited[c_depth][c_index - 1] = True

    def updateAllTree(self, path, reward):
        self.updateRewardTree(path, reward)
        self.updateUvalueTree()
        # Truncate or not
        if path[-1].depth <= np.ceil(
            (np.log(self.rounds) / 2 - np.log(1 / self.nu)) / np.log(1 / self.rho)
        ):
            self.expand(path[-1])
        self.updateBackwardTree()

    def pull(self, time):
        curr_node, self.path = self.optTraverse()
        sample_range = curr_node.get_domain()
        point = []
        for j in range(len(sample_range)):
            # uniformly sample one point, could be replaced by the following
            # x = (sample_range[j][0] + sample_range[j][1]) / 2
            x = np.random.uniform(sample_range[j][0], sample_range[j][1])
            point.append(x)

        return point

    def receive_reward(self, time, reward):
        self.updateAllTree(self.path, reward)
