import pdb

from PyXAB.synthetic_obj import Garland, Himmelblau
from PyXAB.partition.BinaryPartition import BinaryPartition
import numpy.random as random
import matplotlib.pyplot as plt
import numpy as np
from PyXAB.algos.HCT import HCT

# TODO: Change this accordingly


# def visualize_partition():
#
#     target = Garland()
#     support = [[0, 1]]
#     rho = 0.5
#     nu = 1
#
#     rounds = 500
#
#     partition = BinaryPartition(support)
#     algo = HCT(nu=nu, rho=rho, partition=partition)
#
#     textdic = {}
#     plt.axis('off')
#     for i in range(1, rounds+1):
#         point = algo.pull(i)
#         curr_node, path = algo.optTraverse()
#         reward = target.f(point) + random.uniform(-0.1, 0.1)
#         algo.receive_reward(i, reward)
#
#
#         pos = 256.0 / 2 ** (curr_node.depth) + (curr_node.index - 1) * 256.0 / 2 ** (curr_node.depth - 1)
#         plt.scatter(pos, -curr_node.depth)
#
#         if str(curr_node.depth) + ',' + str(curr_node.index) in textdic:
#             textdic[str(curr_node.depth) + ',' + str(curr_node.index)].set_visible(False)
#
#         textvar = plt.text(pos * (1 + 0.01), -curr_node.depth * (1 + 0.01), algo.visitedTimes[curr_node.depth][curr_node.index-1],
#                            fontsize=12)
#
#         # Line connecting the two points
#         y = [-curr_node.depth + 1, -curr_node.depth]
#         parent_pos = 256.0 / 2 ** (curr_node.parent.depth) + (curr_node.parent.index - 1) * 256.0 / 2 ** (curr_node.parent.depth - 1)
#         x = [parent_pos, pos]
#         plt.plot(x, y)
#         textdic[str(curr_node.depth) + ',' + str(curr_node.index)] = textvar
#
#         plt.pause(0.2)
#
#     plt.show()
#
#
#


def visualize_trajectory():
    target = Himmelblau.Himmelblau_Normalized()
    domain = [[-5, 5], [-5, 5]]

    rounds = 500
    partition = BinaryPartition
    algo = HCT(domain=domain, partition=partition)

    cumulative_regret = 0
    cumulative_regret_list = []
    # create the figure

    point_list = []
    reward_set = set()
    for t in range(1, rounds + 1):
        point = algo.pull(t)
        reward = target.f(point) + np.random.uniform(-0.1, 0.1)
        algo.receive_reward(t, reward)
        inst_regret = target.fmax - target.f(point)
        cumulative_regret += inst_regret
        cumulative_regret_list.append(cumulative_regret)

        if point not in point_list and len(point_list) < 18:
            point_list.append(point)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            x = np.linspace(domain[0][0], domain[0][1], 1000)
            y = np.linspace(domain[0][0], domain[0][1], 1000)
            xx, yy = np.meshgrid(x, y)
            z = (-((xx**2 + yy - 11) ** 2) - (xx + yy**2 - 7) ** 2) / 890
            ax.plot_surface(xx, yy, z, alpha=0.4)

            px = np.array([p[0] for p in point_list])
            py = np.array([p[1] for p in point_list])
            pz = np.array([target.f(p) for p in point_list])
            ax.scatter(px, py, pz, s=15, marker="^", color="red", alpha=1)
            for j in range(len(point_list)):
                ax.text(px[j], py[j], pz[j], str(j), fontsize=8)
            if len(point_list) >= 2:
                for j in range(0, len(point_list) - 1):
                    ax.plot(
                        px[j : j + 2],
                        py[j : j + 2],
                        pz[j : j + 2],
                        linewidth=1.5,
                        alpha=1,
                    )

            fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
            plt.savefig("C:/Users/Owner/Downloads/" + str(len(point_list)) + ".png")

            fig = plt.figure()
            ax = fig.add_subplot(111)
            mesh = ax.pcolormesh(xx, yy, z)
            ax.scatter(px, py, s=15, marker="^", color="red", alpha=1)
            for j in range(len(point_list)):
                ax.text(px[j], py[j], str(j), fontsize=8)
            if len(point_list) >= 2:
                for j in range(0, len(point_list) - 1):
                    ax.plot(px[j : j + 2], py[j : j + 2], linewidth=1.5, alpha=1)

            fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
            fig.colorbar(mesh)
            plt.savefig("C:/Users/Owner/Downloads/1-" + str(len(point_list)) + ".png")


visualize_trajectory()
