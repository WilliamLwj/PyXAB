# -*- coding: utf-8 -*-
"""
2-D Example
===================
Run the HCT algorithm on the Himmelblau objective
"""

from PyXAB.synthetic_obj import *
from PyXAB.algos.HOO import T_HOO
from PyXAB.partition.BinaryPartition import BinaryPartition
import matplotlib.pyplot as plt
import numpy as np
from PyXAB.utils.plot import plot_regret




# Define the number of rounds, target, domain, partition, and algorithm
T = 1000
target = Himmelblau.Himmelblau_Normalized()
domain = [[-5, 5], [-5, 5]]
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
plot_regret(np.array(cumulative_regret_list), name='HCT')



# %%
# The following lines of code are only for creating thumbnails and do not need to be used

# sphinx_gallery_thumbnail_number = 2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(domain[0][0], domain[0][1], 1000)
y = np.linspace(domain[0][0], domain[0][1], 1000)
xx, yy = np.meshgrid(x, y)
z = (- (xx ** 2 + yy - 11) ** 2 - (xx + yy ** 2 - 7) ** 2) / 890
ax.plot_surface(xx, yy, z, alpha=0.9)
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)


