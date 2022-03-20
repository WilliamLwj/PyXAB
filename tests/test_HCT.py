from synthetic_obj import *

from algos.HCT import HCT
from partition.BinaryPartition import BinaryPartition
from utils import plot_regret, compare_regret
import numpy as np
import pdb

T = 5000
Target = DoubleSine.DoubleSine()
domain = [[0, 1]]
partition = BinaryPartition(domain)
algo = HCT(partition=partition)

cumulative_regret = 0
cumulative_regret_list = [0]



for t in range(1, T+1):

    # T-HOO
    point = algo.pull(t)
    reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
    algo.receive_reward(t, reward)
    inst_regret = Target.fmax - Target.f(point)
    cumulative_regret += inst_regret
    cumulative_regret_list.append(cumulative_regret)
    #pdb.set_trace()
