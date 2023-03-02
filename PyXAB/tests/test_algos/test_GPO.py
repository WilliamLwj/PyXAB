from PyXAB.synthetic_obj import *

from PyXAB.algos.HCT import HCT
from PyXAB.algos.GPO import GPO
from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import compare_regret
import numpy as np
import pytest


def test_GPO_1():
    algo = HCT
    partition = BinaryPartition
    with pytest.raises(ValueError):
        GPO(partition=partition, algo=algo)



def test_GPO_2():
    algo = HCT
    domain = [[-5, 5], [-5, 5]]
    with pytest.raises(ValueError):
        GPO(domain=domain, algo=algo)

def test_GPO_3():
    domain = [[-5, 5], [-5, 5]]
    partition = BinaryPartition
    with pytest.raises(ValueError):
        GPO(domain=domain, partition=partition)


def test_GPO_4():

    T = 100
    Target = Garland.Garland()
    domain = [[0, 1]]
    partition = BinaryPartition

    algo = GPO(rounds=T, domain=domain, partition=partition, algo=HCT)
    GPO_regret_list = []
    GPO_regret = 0

    for t in range(1, T + 1):
        point = algo.pull(t)
        reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
        algo.receive_reward(t, reward)
        inst_regret = Target.fmax - Target.f(point)
        GPO_regret += inst_regret
        GPO_regret_list.append(GPO_regret)

    # plot the regret
    # regret_dic = {"GPO": np.array(GPO_regret_list)}
    # compare_regret(regret_dic)


