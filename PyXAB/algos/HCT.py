# -*- coding: utf-8 -*-
"""Implementation of HCT (Azar et al, 2014)
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT

# TODO: Make HCT faster

import math
import numpy as np
from PyXAB.algos.Algo import Algorithm
import pdb


def compute_t_plus(x):
    return np.power(2, np.ceil(np.log(x) / np.log(2)))


class HCT(Algorithm):
    """"""

    def __init__(self, nu=1, rho=0.5, delta=0.01, domain=None, partition=None):
        super(HCT, self).__init__()
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        self.partition = partition(domain=domain)

        self.iteration = 1
        self.nu = nu
        self.rho = rho
        self.delta = delta
        self.c = 0.1
        self.c1 = np.power(rho / (3 * nu), 1.0 / 8)

        # List of values that are important

        self.Bvalues = [[np.inf]]
        self.Uvalues = [[np.inf]]
        self.Rewards = [[0]]
        self.visitedTimes = [[0]]
        self.visited = [[True]]
        self.tau_h = [0]  # Threshold on each layer
        self.expand(self.partition.get_root())

    def optTraverse(self):
        """

        Returns
        -------

        """
        # Update the thresholds

        t_plus = compute_t_plus(self.iteration)
        delta_tilde = np.minimum(1.0 / 2, self.c1 * self.delta / t_plus)
        self.tau_h = [0.0]
        for i in range(1, self.partition.get_depth() + 1):
            self.tau_h.append(
                np.ceil(
                    self.c**2
                    * math.log(1 / delta_tilde)
                    * self.rho ** (-2 * i)
                    / self.nu**2
                )
            )

        curr_node = self.partition.get_root()
        path = [curr_node]

        while (
            self.visitedTimes[curr_node.get_depth()][curr_node.get_index() - 1]
            >= self.tau_h[curr_node.get_depth()]
            and curr_node.get_children() is not None
        ):
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
        node = path[-1]
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
        t_plus = compute_t_plus(self.iteration)
        delta_tilde = np.minimum(1, self.c1 * self.delta / t_plus)
        node_list = self.partition.get_node_list()
        for layer in node_list:
            for node in layer:
                depth = node.get_depth()
                index = node.get_index()

                if self.visitedTimes[depth][index - 1] == 0:
                    continue
                else:
                    UCB = math.sqrt(
                        self.c**2
                        * math.log(1 / delta_tilde)
                        / self.visitedTimes[depth][index - 1]
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
            raise ValueError
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
            raise ValueError
        else:
            for child in children:
                c_depth = child.get_depth()
                c_index = child.get_index()
                self.visited[c_depth][c_index - 1] = True

    def updateAllTree(self, path, end_node, reward):
        t_plus = compute_t_plus(self.iteration)
        delta_tilde = np.minimum(1, self.c1 * self.delta / t_plus)

        if self.iteration == compute_t_plus(self.iteration):
            self.updateUvalueTree()
            self.updateBackwardTree()

        path.append(end_node)

        self.updateRewardTree(path, reward)

        end_node = path[-1]
        en_depth = end_node.get_depth()
        en_index = end_node.get_index()

        self.Uvalues[en_depth][en_index - 1] = (
            self.Rewards[en_depth][en_index - 1]
            + math.sqrt(
                self.c**2
                * math.log(1 / delta_tilde)
                / self.visitedTimes[en_depth][en_index - 1]
            )
            + self.nu * (self.rho**end_node.depth)
        )

        self.updateBackwardTree()

        if self.visitedTimes[en_depth][en_index - 1] >= self.tau_h[en_depth]:
            self.expand(end_node)

    def pull(self, time):
        self.curr_node, self.path = self.optTraverse()
        sample_range = self.curr_node.get_domain()
        point = []
        for j in range(len(sample_range)):
            x = (sample_range[j][0] + sample_range[j][1]) / 2
            point.append(x)

        return point

    def receive_reward(self, time, reward):
        self.updateAllTree(self.path, self.curr_node, reward)
