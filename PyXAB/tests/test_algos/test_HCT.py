from PyXAB.synthetic_obj import *

from PyXAB.algos.HCT import HCT
from PyXAB.partition.BinaryPartition import BinaryPartition
import numpy as np
from PyXAB.utils.plot import plot_regret, compare_regret_withsd, compare_regret
import pytest

def test_HCT_value_error_1():

    partition = BinaryPartition
    with pytest.raises(ValueError):
        HCT( partition=partition)

def test_HCT_value_error_2():

    domain = [[0, 1]]
    with pytest.raises(ValueError):
        HCT(domain=domain)

def test_HCT_initialization():

    partition = BinaryPartition
    domain = [[0, 1]]
    algo = HCT(domain=domain, partition=partition)
    root = algo.partition.get_root()
    assert root.get_mean_reward() == 0
    assert root.get_visited_times() == 0

def test_HCT_Garland():
    T = 100
    Target = Garland.Garland()
    domain = [[0, 1]]
    partition = BinaryPartition
    algo = HCT( domain=domain, partition=partition)

    cumulative_regret = 0
    cumulative_regret_list = [0]

    for t in range(1, T + 1):
        print(t)
        point = algo.pull(t)
        reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
        algo.receive_reward(t, reward)
        inst_regret = Target.fmax - Target.f(point)
        cumulative_regret += inst_regret
        cumulative_regret_list.append(cumulative_regret)

        print("T-HOO: ", point)


def test_HCT_Himmelblau():
    T = 100
    Target = Himmelblau.Himmelblau_Normalized()
    domain = [[-5, 5], [-5, 5]]
    partition = BinaryPartition
    algo = HCT(domain=domain, partition=partition)

    cumulative_regret = 0
    cumulative_regret_list = [0]

    for t in range(1, T + 1):
        print(t)
        point = algo.pull(t)
        reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
        algo.receive_reward(t, reward)
        inst_regret = Target.fmax - Target.f(point)
        cumulative_regret += inst_regret
        cumulative_regret_list.append(cumulative_regret)

        print("T-HOO: ", point)