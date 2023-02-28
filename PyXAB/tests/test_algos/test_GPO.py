from PyXAB.synthetic_obj import *

from PyXAB.algos.HCT import HCT
from PyXAB.algos.GPO import GPO
from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import compare_regret
import numpy as np

T = 100
Target = Garland.Garland()
domain = [[0, 1]]
partition = BinaryPartition


GPO = GPO(rounds=T, domain=domain, partition=partition, algo=HCT)
GPO_regret_list = []
GPO_regret = 0


for t in range(1, T + 1):
    point = GPO.pull(t)
    reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
    GPO.receive_reward(t, reward)
    inst_regret = Target.fmax - Target.f(point)
    GPO_regret += inst_regret
    GPO_regret_list.append(GPO_regret)

# plot the regret
# regret_dic = {"GPO": np.array(GPO_regret_list)}
# compare_regret(regret_dic)
