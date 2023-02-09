from PyXAB.synthetic_obj import *

from PyXAB.algos.StroquOOL import StroquOOL
from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import compare_regret
import math
import numpy as np
import pdb





T = 1000
H = math.floor(T / (2 * (np.log2(T) + 1)**2))
Target = Garland.Garland()
domain = [[0, 1]]
partition = BinaryPartition
algo = StroquOOL(n = T, domain=domain, partition=partition)
node_list = algo.partition.get_node_list()
algo.partition.make_children(node_list[0][0], newlayer=True)

for i in range(H):
    for j in range(len(algo.partition.root.get_children())):
        node = algo.partition.root.get_children()[j]
        reward = Target.f(node.get_cpoint()) + np.random.uniform(-0.1, 0.1)
        algo.receive_reward(node, reward)


for h in range(1, H):
    node_list = algo.pull(h) # nodes at depth = h to open
    for i in range(len(node_list)):
        node = node_list[i][0]
        open_time = node_list[i][1]
        # index = point_list[i][2]
        for p in range(2**open_time):
            reward = Target.f(node.get_cpoint()) + np.random.uniform(-0.1, 0.1)
            algo.receive_reward(node, reward)
    
    
# Cross Validation

chosen = algo.get_chosen()
p_max = algo.p_max
max_point = None
max_value = -np.inf

for p in range(p_max + 1):
    for i in range(len(chosen)):
        # print(chosen[i].get_visited_times())
        if chosen[i].get_visited_times() >= 2**p:
            chosen[i].remove_reward
            point = chosen[i].get_cpoint()
            for h in range(H):
                reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
                chosen[i].update_reward(reward)
        
        chosen[i].compute_mean_reward()
        if chosen[i].get_mean_reward() >= max_value:
            max_point = point

print(Target.fmax - Target.f(max_point))
        
    