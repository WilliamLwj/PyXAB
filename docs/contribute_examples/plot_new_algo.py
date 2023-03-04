# -*- coding: utf-8 -*-
"""
New Algorithm
=====================
A (dummy) example to implement a new PyXAB algorithm. First import all the useful packages
"""

from PyXAB.synthetic_obj.Garland import Garland
from PyXAB.algos.Algo import Algorithm
from PyXAB.partition.BinaryPartition import BinaryPartition
import numpy as np


# %%
# Now let us suppose that we want to implement a new (dummy) algorithm that always select all nodes on each layer
# and then evaluate them for a number of times

class Dummy(Algorithm):
    def __init__(self, evaluation=5, rounds=1000, domain=None, partition=None):
        super(Dummy, self).__init__()
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        self.partition = partition(domain=domain)
        self.iteration = 0
        self.evaluation = evaluation
        self.rounds = rounds
        self.partition.deepen()

        self.iterator = 0

    # we need to re-write the pull function
    def pull(self, time):

        depth = self.partition.get_depth()
        index = self.iterator // self.evaluation
        nodes = self.partition.get_node_list()
        point = nodes[depth][index].get_cpoint()
        self.iterator += 1                  # get a point and increase the iterator
        if self.iterator >= self.evaluation * len(nodes[depth]):     # If every point is evaluated, deepen the partition
            self.partition.deepen()

        return point

    # we need to re-write the receive_reward function
    def receive_reward(self, time, reward):
        # No update given the reward for the dummy algorithm
        pass


# %%
# Define the number of rounds, the target, the domain, the partition, and the algorithm for the learning process
T = 100
target = Garland()
domain = [[0, 1]]
partition = BinaryPartition
algo = Dummy(rounds=T, domain=domain, partition=partition)        # The new algorithm


# %%
# As shown below, the Dummy algorithm now optimizes the objective

for t in range(1, T+1):

    point = algo.pull(t)
    reward = target.f(point) + np.random.uniform(-0.1, 0.1)     # uniform noise
    algo.receive_reward(t, reward)
    #print(point)