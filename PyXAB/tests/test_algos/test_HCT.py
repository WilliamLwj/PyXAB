from PyXAB.synthetic_obj import *

from PyXAB.algos.HCT import HCT
from PyXAB.partition.BinaryPartition import BinaryPartition
import numpy as np
from PyXAB.utils.plot import plot_regret


T = 100
target = Himmelblau.Himmelblau()
domain = [[-5, 5], [-5, 5]]
partition = BinaryPartition
algo = HCT(domain=domain, partition=partition)

cumulative_regret = 0
cumulative_regret_list = []


## uniform noise

for t in range(1, T + 1):
    point = algo.pull(t)
    reward = target.f(point) + np.random.uniform(-0.1, 0.1)
    algo.receive_reward(t, reward)
    inst_regret = target.fmax - target.f(point)
    cumulative_regret += inst_regret
    cumulative_regret_list.append(cumulative_regret)

# plot_regret(np.array(cumulative_regret_list))
