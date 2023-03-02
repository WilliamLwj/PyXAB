from PyXAB.synthetic_obj import *

from PyXAB.algos.StoSOO import StoSOO
from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import compare_regret
import numpy as np
import pytest
import pdb

def test_StoSOO_1():
    T = 1000
    partition = BinaryPartition
    with pytest.raises(ValueError):
        StoSOO(n=T, partition=partition)

def test_StoSOO_2():
    T = 1000
    domain = [[0, 1]]
    with pytest.raises(ValueError):
        StoSOO(n=T, domain=domain)

def test_StoSOO_3():
    T = 100
    Target = Garland.Garland()
    domain = [[0, 1]]
    partition = BinaryPartition
    algo = StoSOO(n=T, domain=domain, partition=partition)

    regret = 0
    StoSOO_regret_list = []

    for t in range(1, T + 1):
        # T-HOO
        point = algo.pull(t)
        reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
        algo.receive_reward(t, reward)
        inst_regret = Target.fmax - Target.f(point)
        regret += inst_regret
        StoSOO_regret_list.append(regret)

    last_point = algo.get_last_point()
    print(T, Target.fmax - Target.f(last_point))
# plot the regret
# regret_dic = {"StoSOO": np.array(StoSOO_regret_list)}
# compare_regret(regret_dic)
