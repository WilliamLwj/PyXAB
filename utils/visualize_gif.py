from synthetic_obj.Garland import Garland
from partition.BinaryPartition import BinaryPartition
import numpy.random as random
import matplotlib.pyplot as plt


import os
import pdb
import numpy as np
from algos.HCT import HCT

def visualize_HCT():

    target = Garland()
    support = [[0, 1]]
    rho = 0.5
    nu = 1

    rounds = 500

    partition = BinaryPartition(support)
    algo = HCT(nu=nu, rho=rho, partition=partition)

    textdic = {}
    plt.axis('off')
    for i in range(1, rounds+1):
        point = algo.pull(i)
        curr_node, path = algo.optTraverse()
        reward = target.f(point) + random.uniform(-0.1, 0.1)
        algo.receive_reward(i, reward)


        pos = 256.0 / 2 ** (curr_node.depth) + (curr_node.index - 1) * 256.0 / 2 ** (curr_node.depth - 1)
        plt.scatter(pos, -curr_node.depth)

        if str(curr_node.depth) + ',' + str(curr_node.index) in textdic:
            textdic[str(curr_node.depth) + ',' + str(curr_node.index)].set_visible(False)

        textvar = plt.text(pos * (1 + 0.01), -curr_node.depth * (1 + 0.01), algo.visitedTimes[curr_node.depth][curr_node.index-1],
                           fontsize=12)

        # Line connecting the two points
        y = [-curr_node.depth + 1, -curr_node.depth]
        parent_pos = 256.0 / 2 ** (curr_node.parent.depth) + (curr_node.parent.index - 1) * 256.0 / 2 ** (curr_node.parent.depth - 1)
        x = [parent_pos, pos]
        plt.plot(x, y)
        textdic[str(curr_node.depth) + ',' + str(curr_node.index)] = textvar

        plt.pause(0.2)

    plt.show()


visualize_HCT()