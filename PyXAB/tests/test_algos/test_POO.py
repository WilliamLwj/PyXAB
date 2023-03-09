from PyXAB.synthetic_obj import *

from PyXAB.algos.HOO import T_HOO
from PyXAB.algos.POO import POO
from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import compare_regret
import numpy as np
import pytest


def test_POO_1():
    algo = T_HOO
    partition = BinaryPartition
    with pytest.raises(ValueError):
        algo = POO(partition=partition, algo=algo)


def test_POO_2():
    algo = T_HOO
    domain = [[0, 1]]
    with pytest.raises(ValueError):
        algo = POO(domain=domain, algo=algo)


def test_POO_3():
    domain = [[0, 1]]
    partition = BinaryPartition
    with pytest.raises(ValueError):
        algo = POO(domain=domain, partition=partition)


def test_POO_4():
    T = 100
    Target = Garland.Garland()
    domain = [[0, 1]]
    partition = BinaryPartition

    algo = POO(rounds=T, domain=domain, partition=partition, algo=T_HOO)
    POO_regret_list = []
    POO_regret = 0

    for t in range(1, T + 1):
        point = algo.pull(t)
        reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
        algo.receive_reward(t, reward)
        inst_regret = Target.fmax - Target.f(point)
        POO_regret += inst_regret
        POO_regret_list.append(POO_regret)

    algo.get_last_point()
    # plot the regret
    # regret_dic = {"POO": np.array(POO_regret_list)}
    # compare_regret(regret_dic)
