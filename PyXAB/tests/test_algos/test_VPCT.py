from PyXAB.synthetic_obj import *

from PyXAB.algos.VPCT import VPCT
from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import compare_regret
import numpy as np
import pytest


def test_VPCT_value_error_1():
    partition = BinaryPartition
    with pytest.raises(ValueError):
        VPCT(partition=partition)


def test_VPCT_value_error_2():
    domain = [[-5, 5], [-5, 5]]
    with pytest.raises(ValueError):
        VPCT(domain=domain)


def test_VPCT_Garland():
    T = 1000
    Target = Garland.Garland()
    domain = [[0, 1]]
    partition = BinaryPartition
    algo = VPCT(rounds=T, domain=domain, partition=partition)

    VPCT_regret_list = []
    regret = 0

    for t in range(1, T + 1):
        point = algo.pull(t)
        reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
        algo.receive_reward(t, reward)
        inst_regret = Target.fmax - Target.f(point)
        regret += inst_regret
        VPCT_regret_list.append(regret)

    print('VCPT: ', algo.get_last_point())

def test_VPCT_Himmelblau():
    T = 1000
    target = Himmelblau.Himmelblau()
    domain = [[-5, 5], [-5, 5]]
    partition = BinaryPartition
    algo = VPCT(rounds=T, domain=domain, partition=partition)

    cumulative_regret = 0
    cumulative_regret_list = []

    ## uniform noise

    for t in range(1, T + 1):
        point = algo.pull(t)
        reward = target.f(point) + np.random.uniform(-0.1, 0.1)
        algo.receive_reward(t, reward)
        inst_regret = target.fmax - target.f(point)
        cumulative_regret += inst_regret
        cumulative_regret_list.append(cumulative_regret)

    print('VPCT: ', algo.get_last_point())

