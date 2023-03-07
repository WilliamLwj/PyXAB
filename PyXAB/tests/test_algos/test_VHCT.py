from PyXAB.synthetic_obj import *

from PyXAB.algos.VHCT import VHCT

from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import compare_regret
import numpy as np
import pytest


def test_VHCT_value_error_1():

    partition = BinaryPartition
    with pytest.raises(ValueError):
        VHCT( partition=partition)

def test_VHCT_value_error_2():

    domain = [[0, 1]]
    with pytest.raises(ValueError):
        VHCT(domain=domain)

def test_VHCT_initialization():

    partition = BinaryPartition
    domain = [[0, 1]]
    algo = VHCT(domain=domain, partition=partition)
    root = algo.partition.get_root()
    assert root.get_mean_reward() == 0
    assert root.get_visited_times() == 0

def test_VHCT_Garland():
    T = 100
    Target = Garland.Garland()
    domain = [[0, 1]]
    partition = BinaryPartition
    algo = VHCT(domain=domain, partition=partition)

    cumulative_regret = 0
    cumulative_regret_list = [0]

    for t in range(1, T + 1):
        # VHCT
        point = algo.pull(t)
        reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
        algo.receive_reward(t, reward)
        inst_regret = Target.fmax - Target.f(point)
        cumulative_regret += inst_regret
        cumulative_regret_list.append(cumulative_regret)


def test_VHCT_Himmelblau():
    T = 100
    Target = Himmelblau.Himmelblau_Normalized()
    domain = [[-5, 5], [-5, 5]]
    partition = BinaryPartition
    algo = VHCT(domain=domain, partition=partition)

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