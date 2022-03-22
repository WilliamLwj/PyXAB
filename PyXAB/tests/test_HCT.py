from PyXAB.synthetic_obj import *

from PyXAB.algos.HCT import HCT
from PyXAB.partition.BinaryPartition import BinaryPartition
import numpy as np
from PyXAB.utils.plot import plot_regret


T = 1000
Target = HimmelBlau.Himmelblau()
domain = [[-5, 5], [-5, 5]]
partition = BinaryPartition
algo = HCT(domain=domain, partition=partition)

cumulative_regret = 0
cumulative_regret_list = [0]



for t in range(1, T+1):

    # HCT
    print(t)
    point = algo.pull(t)
    reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
    algo.receive_reward(t, reward)
    inst_regret = Target.fmax - Target.f(point)
    cumulative_regret += inst_regret
    cumulative_regret_list.append(cumulative_regret)
    #pdb.set_trace()

plot_regret(np.array(cumulative_regret_list))