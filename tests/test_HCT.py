from synthetic_obj import *

from algos.HCT import HCT, HCT_tree
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


HCT_regret_list = []
regret = 0

tree = HCT_tree(1, 0.75, domain)

for t in range(1, T+1):

    # T-HOO
    point = algo.pull(t)
    reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
    algo.receive_reward(t, reward)
    inst_regret = Target.fmax - Target.f(point)
    cumulative_regret += inst_regret
    cumulative_regret_list.append(cumulative_regret)

    curr_node, path = tree.optTraverse()
    sample_range = curr_node.range
    pulled_x = []
    for j in range(len(sample_range)):
        x = (sample_range[j][0] + sample_range[j][1]) / 2.0
        pulled_x.append(x)
    reward = Target.f(pulled_x) + np.random.uniform(-0.1, 0.1)
    tree.updateAllTree(path, curr_node, reward)
    simple_regret = Target.fmax - Target.f(pulled_x)
    regret += simple_regret
    HCT_regret_list.append(regret)

    #pdb.set_trace()

regret_dic = {'HCT': np.array(cumulative_regret_list),
              'old_HCT': np.array(HCT_regret_list)}
compare_regret(regret_dic)