from synthetic_obj.Garland import Garland
import numpy.random as random
import matplotlib.pyplot as plt
from algos.HCT import HCT_tree

import os
import pdb
import numpy as np

def visualize_HCT():

    Target = Garland()
    support = [[0, 1]]
    rho = 0.5
    nu = 1
    rounds = 500

    noise = random.uniform(-0.1, 0.1)

    tree = HCT_tree(nu, rho, support)

    HCT_regret_list = []

    regret = 0
    textdic = {}
    plt.axis('off')
    for i in range(rounds):
        curr_node, path = tree.optTraverse()
        sample_range = curr_node.range
        pulled_x = []
        for j in range(len(sample_range)):
            x = (sample_range[j][0] + sample_range[j][1]) / 2.0
            pulled_x.append(x)
        reward = Target.f(pulled_x) + noise
        tree.updateAllTree(path, curr_node, reward)

        simple_regret = Target.fmax - Target.f(pulled_x)
        regret += simple_regret
        HCT_regret_list.append(regret / (i + 1))
        pos = 256.0 / 2 ** (curr_node.depth) + (curr_node.index - 1) * 256.0 / 2 ** (curr_node.depth - 1)
        plt.scatter(pos, -curr_node.depth)

        if str(curr_node.depth) + ',' + str(curr_node.index) in textdic:
            textdic[str(curr_node.depth) + ',' + str(curr_node.index)].set_visible(False)

        textvar = plt.text(pos * (1 + 0.01), -curr_node.depth * (1 + 0.01), curr_node.visitedTimes,
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




