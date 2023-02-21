import numpy.random as random
import matplotlib.pyplot as plt


import os
import pdb
import numpy as np


def plot_regret(regret_list, name="T-HOO"):
    x = np.arange(regret_list.shape[0])
    plt.plot(x, regret_list, linewidth=2, label=name, alpha=0.9)
    plt.legend(loc="upper right", prop={"size": 14})
    plt.show()


def compare_regret(regret_dic):
    for name in regret_dic.keys():
        regret_list = regret_dic[name]
        x = np.arange(regret_list.shape[0])
        plt.plot(x, regret_list, linewidth=2, label=name, alpha=0.9)

    plt.legend(loc="upper right", prop={"size": 14})
    plt.show()


def compare_regret_withsd(dictionary, x_range=None, y_range=None):
    regret = dictionary["regret"]
    colors = dictionary["colors"]
    labels = dictionary["labels"]

    plt.figure(figsize=(6, 5), dpi=100)

    for i in range(len(regret)):
        regret_array = regret[i]
        x = np.arange((regret_array.shape[1])) + 1
        mean = np.mean(regret_array, axis=0)
        std = np.std(regret_array, axis=0)

        plt.plot(x, mean, linewidth=2, color=colors[i], label=labels[i], alpha=0.9)
        plt.fill_between(x, mean + 1 * std, mean - 1 * std, color=colors[i], alpha=0.3)

    plt.legend(loc="upper left", prop={"size": 16})
    plt.xlabel("Rounds", fontsize=16)
    if x_range is not None:
        plt.xlim(x_range)
    if y_range is not None:
        plt.ylim(y_range)
    plt.ylabel("Cumulative Regret", fontsize=16)
    plt.show()
