# -*- coding: utf-8 -*-
"""
1-D Example
================
Run the T_HOO algorithm on the Garland objective
"""

from PyXAB.synthetic_obj import *
from PyXAB.algos.HOO import T_HOO
from PyXAB.partition.BinaryPartition import BinaryPartition
import matplotlib.pyplot as plt
import numpy as np
from PyXAB.utils.plot import plot_regret




# Define the number of rounds, target, domain, partition, and algorithm
T = 1000
target = Garland.Garland()
domain = [[0, 1]]
partition = BinaryPartition
algo = T_HOO(domain=domain, partition=partition)


# regret and regret list
cumulative_regret = 0
cumulative_regret_list = []


# uniform noise
for t in range(1, T+1):

    point = algo.pull(t)
    reward = target.f(point) + np.random.uniform(-0.1, 0.1)
    algo.receive_reward(t, reward)
    inst_regret = target.fmax - target.f(point)
    cumulative_regret += inst_regret
    cumulative_regret_list.append(cumulative_regret)


# plot the regret
plot_regret(np.array(cumulative_regret_list), name='HOO')





# %%
# The following lines of code are only for creating thumbnails and do not need to be used

# sphinx_gallery_thumbnail_number = 2
fig = plt.figure()
ax = fig.add_subplot(111)

x = np.linspace(domain[0][0], domain[0][1], 1000)
z = x * (1-x) * (4 - np.sqrt(np.abs(np.sin(60 * x))))
ax.plot(x, z, alpha=0.9)
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
