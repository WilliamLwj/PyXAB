# -*- coding: utf-8 -*-
"""
New Objective
=====================
An example to implement a new blackbox objective for any PyXAB algorithm to optimize. First import all the useful packages
"""

from PyXAB.synthetic_obj.Objective import Objective
from PyXAB.algos.HOO import T_HOO
from PyXAB.partition.BinaryPartition import BinaryPartition


# %%
# We have already shown a ``Sine`` function example in the general instructions.
# Here, we give another very simple example. Suppose the objective is only a constant, i.e., ``f(x) = 1`` everywhere,
# then the class definition would be as simple as.

class Constant_1(Objective):
    def __init__(self):
        self.fmax = 1

    def f(self, x):
        return 1



# %%
# The inheritance of the ``Objective`` class is unnecessary (but highly recommended for consistency).
# E.g., another possible definition could be

class Constant_2():

    def evaluate(self, x):
        return 1


# %%
# Let us suppose the domain for this objective is ``[0, 10]``, and we use the binary partition for the domain and the HOO
# algorithm to optimize the objectives

T = 10
target1 = Constant_1()
target2 = Constant_2()
domain = [[0, 10]]
partition = BinaryPartition
algo = T_HOO(domain=domain, partition=partition)

# %%
# Now as can be seen below, the objectives are ready to be optimized

# Optimize Objective Constant_1
for t in range(1, T+1):

    point = algo.pull(t)
    reward = target1.f(point)
    algo.receive_reward(t, reward)


# Optimize Objective Constant_2
for t in range(1, T+1):

    point = algo.pull(t)
    reward = target2.evaluate(point)
    algo.receive_reward(t, reward)
