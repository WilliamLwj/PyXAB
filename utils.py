from synthetic_obj import DifficultFunc, DoubleSine, Garland, Himmelblau, Ackley, Rastrigin
import numpy.random as random
import matplotlib.pyplot as plt
import os
import pdb

import numpy as np
Target3 = Garland()

def visualize_HCT():

    from algos.HCT import HCT_tree

    Target = Target3
    support = [[0, 1]]
    rho = 0.5
    nu = 1

    rounds = 500

    noise = random.uniform(-0.1, 0.1)

    HCT_tree = HCT_tree(nu, rho, support)

    HCT_regret_list = []
    regret = 0
    textdic = {}
    plt.axis('off')
    for i in range(rounds):
        curr_node, path = HCT_tree.optTraverse()
        sample_range = curr_node.range
        pulled_x = []
        for j in range(len(sample_range)):
            x = (sample_range[j][0] + sample_range[j][1]) / 2.0
            pulled_x.append(x)
        reward = Target.f(pulled_x) + noise
        HCT_tree.updateAllTree(path, curr_node, reward)

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

def plot_collaboration():


    import matplotlib.pyplot as plt
    cont_x = np.arange(1, 7, 1/100)
    disc_x = np.arange(1, 6)
    OE = (0.5)**cont_x
    plt.plot(cont_x, OE, color='red', label='OE$_h$', alpha=0.9, linewidth=3, linestyle='--')
    plt.plot(cont_x, -OE, color='red', alpha=0.2, linewidth=3, linestyle='--')
    plt.scatter(disc_x, 0.5 ** disc_x, color='red',  linewidth=2)

    plt.plot(cont_x, np.zeros(cont_x.shape), color='black', linewidth = 1, linestyle='-.')

    for i in disc_x:
        x = np.array([i-0.1, i, i+0.1])
        y = np.array([i+0.1, i+0.1, i+0.1])
        if i < 5:
          plt.fill_between(x, 0.5 ** y, -0.5 ** y, color='yellow', alpha=0.7, label='SE$_{layer}$'.format(layer=i))
        else:
          plt.fill_between(x, 0.5 ** y + 0.05, -0.5 ** y -0.05, color='red', alpha=0.7, label='SE$_{layer}$'.format(layer=i))
    plt.xlim((0.8, 6))
    plt.legend(loc='upper right', prop={'size': 12}, ncol=2)
    plt.xlabel('Layer number h')
    plt.ylabel('Dynamics of OE and SE')
    plt.text(0.8, -0.5, 'SE$_1$ < OE$_1$', fontsize=12, color='black')
    plt.text(1.6, -0.35, 'SE$_2$ < OE$_2$', fontsize=12, color='black')
    plt.text(2.6, -0.2, 'SE$_3$ < OE$_3$', fontsize=12, color='black')
    plt.text(3.6, -0.15, 'SE$_4$ < OE$_4$', fontsize=12, color='black')
    plt.text(4.6, -0.15, 'SE$_5$ > OE$_5$', fontsize=12, color='red')
    plt.show()
    #print(np.max(np.array(val)))

#plot_collaboration()




