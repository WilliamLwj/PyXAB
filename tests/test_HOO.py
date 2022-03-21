from synthetic_obj import *

from algos.HOO import T_HOO
from partition.BinaryPartition import BinaryPartition
from utils.plot import compare_regret
import numpy as np

T = 5000
Target = Garland.Garland()
domain = [[0, 1]]
partition = BinaryPartition(domain)
algo = T_HOO(rounds=T, partition=partition)

cumulative_regret = 0
cumulative_regret_list = [0]


HOO_regret_list = []
regret = 0


for t in range(T):

    # T-HOO
    point = algo.pull(t)
    reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
    algo.receive_reward(t, reward)
    inst_regret = Target.fmax - Target.f(point)
    cumulative_regret += inst_regret
    cumulative_regret_list.append(cumulative_regret)

    print('T-HOO: ', point)

    #pdb.set_trace()

regret_dic = {'T-HOO': np.array(cumulative_regret_list),
              'HOO': np.array(HOO_regret_list)}
compare_regret(regret_dic)