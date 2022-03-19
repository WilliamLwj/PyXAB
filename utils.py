from synthetic_obj.Garland import Garland
import numpy.random as random
import matplotlib.pyplot as plt
from algos.HCT import HCT_tree

import os
import pdb
import numpy as np


def plot_regret(regret_list, name='T-HOO'):

    x = np.arange(regret_list.shape[0])
    plt.plot(x, regret_list, linewidth=2, label=name, alpha=0.9)
    plt.legend(loc='upper right', prop={'size': 14})
    plt.show()




def compare_regret(regret_dic):

    for name in regret_dic.keys():
        regret_list = regret_dic[name]
        x = np.arange(regret_list.shape[0])
        plt.plot(x, regret_list, linewidth=2, label=name, alpha=0.9)

    plt.legend(loc='upper right', prop={'size': 14})
    plt.show()
