from PyXAB.synthetic_obj import *

from PyXAB.algos.HOO import T_HOO
from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import compare_regret
import numpy as np
import pytest

def test_HOO_value_error_1():
    T = 1000
    partition = BinaryPartition
    with pytest.raises(ValueError):
        T_HOO(rounds=T, partition=partition)

def test_HOO_value_error_2():
    T = 1000
    domain = [[0, 1]]
    with pytest.raises(ValueError):
        T_HOO(rounds=T, domain=domain)

def test_HOO_initialization():
    T = 1000
    partition = BinaryPartition
    domain = [[0, 1]]
    algo = T_HOO(rounds=T, domain=domain, partition=partition)
    root = algo.partition.get_root()
    assert root.get_mean_reward() == 0
    assert root.get_visited_times() == 0

def test_HOO_Garland():
    T = 100
    Target = Garland.Garland()
    domain = [[0, 1]]
    partition = BinaryPartition
    algo = T_HOO(rounds=T, domain=domain, partition=partition)

    cumulative_regret = 0
    cumulative_regret_list = [0]

    for t in range(1, T + 1):
        # T-HOO
        print(t)
        point = algo.pull(t)
        reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
        algo.receive_reward(t, reward)
        inst_regret = Target.fmax - Target.f(point)
        cumulative_regret += inst_regret
        cumulative_regret_list.append(cumulative_regret)

        print("T-HOO: ", point)

    # plot the result
    # regret_dic = {
    #     "T-HOO": np.array(cumulative_regret_list),
    # }
    # compare_regret(regret_dic)
