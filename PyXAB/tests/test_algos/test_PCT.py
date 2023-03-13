from PyXAB.synthetic_obj import *

from PyXAB.algos.PCT import PCT
from PyXAB.partition.BinaryPartition import BinaryPartition
import numpy as np
from PyXAB.utils.plot import plot_regret
import pytest


def test_PCT_value_error_1():
    partition = BinaryPartition
    with pytest.raises(ValueError):
        algo = PCT(partition=partition)


def test_PCT_value_error_2():
    domain = [[-5, 5], [-5, 5]]
    with pytest.raises(ValueError):
        algo = PCT(domain=domain)

def test_PCT_Garland():
    T = 1000
    target = Garland.Garland()
    domain = [[0, 1]]
    partition = BinaryPartition
    algo = PCT(rounds=T, domain=domain, partition=partition)

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

    # plot_regret(np.array(cumulative_regret_list))

    print('PCT: ', algo.get_last_point())

def test_PCT_Himmelblau():
    T = 1000
    target = Himmelblau.Himmelblau()
    domain = [[-5, 5], [-5, 5]]
    partition = BinaryPartition
    algo = PCT(rounds=T, domain=domain, partition=partition)

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

    # plot_regret(np.array(cumulative_regret_list))

    print('PCT: ', algo.get_last_point())