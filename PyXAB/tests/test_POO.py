from PyXAB.synthetic_obj import *

from PyXAB.algos.HOO import T_HOO
from PyXAB.algos.POO import POO
from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import compare_regret
import numpy as np

T = 1000
Target = Garland.Garland()
domain = [[0, 1]]
partition = BinaryPartition
algo = T_HOO(domain=domain, partition=partition)


T_HOO_regret_list = []
regret = 0


POO = POO(rounds=T, domain=domain, partition=partition, algo=T_HOO)
POO_regret_list = []
POO_regret = 0


for t in range(1, T + 1):

    # T-HOO
    point = algo.pull(t)
    reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
    algo.receive_reward(t, reward)
    inst_regret = Target.fmax - Target.f(point)
    regret += inst_regret
    T_HOO_regret_list.append(regret)

    print(t)

    point = POO.pull(t)
    reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
    POO.receive_reward(t, reward)
    inst_regret = Target.fmax - Target.f(point)
    POO_regret += inst_regret
    POO_regret_list.append(POO_regret)


regret_dic = {"T_HOO": np.array(T_HOO_regret_list), "POO": np.array(POO_regret_list)}

compare_regret(regret_dic)
