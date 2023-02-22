from PyXAB.synthetic_obj import *

from PyXAB.algos.VHCT import VHCT

from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import compare_regret
import numpy as np

T = 100
Target = DoubleSine.DoubleSine()
domain = [[0, 1]]
partition = BinaryPartition
algo = VHCT(domain=domain, partition=partition)

cumulative_regret = 0
cumulative_regret_list = [0]


for t in range(1, T + 1):
    # T-HOO
    point = algo.pull(t)
    reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
    algo.receive_reward(t, reward)
    inst_regret = Target.fmax - Target.f(point)
    cumulative_regret += inst_regret
    cumulative_regret_list.append(cumulative_regret)

    print("t: ", t, "VHCT: ", point)


# plot regret
# regret_dic = {
#     "VHCT": np.array(cumulative_regret_list),
# }
# compare_regret(regret_dic)
