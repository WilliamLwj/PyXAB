from PyXAB.synthetic_obj import *

from PyXAB.algos.Zooming import Zooming

from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import compare_regret
import numpy as np
import pytest


def test_Zooming_value_error_1():

    partition = BinaryPartition
    with pytest.raises(ValueError):
        Zooming(partition=partition)


def test_Zooming_value_error_2():

    domain = [[0, 1]]
    with pytest.raises(ValueError):
        Zooming(domain=domain)


def test_Zooming_Garland():
    T = 100
    Target = Garland.Garland()
    domain = [[0, 1]]
    partition = BinaryPartition
    algo = Zooming(domain=domain, partition=partition)

    cumulative_regret = 0
    cumulative_regret_list = [0]

    for t in range(1, T + 1):
        # Zooming
        point = algo.pull(t)
        reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
        algo.receive_reward(t, reward)
        inst_regret = Target.fmax - Target.f(point)
        cumulative_regret += inst_regret
        cumulative_regret_list.append(cumulative_regret)

    compare_regret({"Zooming": np.array(cumulative_regret_list)})
    print("Zooming: ", algo.get_last_point())


def test_Zooming_Himmelblau():
    T = 100
    Target = Himmelblau.Himmelblau_Normalized()
    domain = [[-5, 5], [-5, 5]]
    partition = BinaryPartition
    algo = Zooming(domain=domain, partition=partition)

    cumulative_regret = 0
    cumulative_regret_list = [0]

    for t in range(1, T + 1):
        point = algo.pull(t)
        reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
        algo.receive_reward(t, reward)
        inst_regret = Target.fmax - Target.f(point)
        cumulative_regret += inst_regret
        cumulative_regret_list.append(cumulative_regret)

    compare_regret({"Zooming": np.array(cumulative_regret_list)})
    print("Zooming: ", algo.get_last_point())
