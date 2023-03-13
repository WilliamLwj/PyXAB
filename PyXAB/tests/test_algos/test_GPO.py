from PyXAB.synthetic_obj import *

from PyXAB.algos.HCT import HCT
from PyXAB.algos.HOO import T_HOO
from PyXAB.algos.VHCT import VHCT
from PyXAB.algos.POO import POO
from PyXAB.algos.StoSOO import StoSOO
from PyXAB.algos.GPO import GPO
from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import compare_regret
import numpy as np
import pytest


def test_GPO_value_error_1():
    # no domain
    algo = HCT
    partition = BinaryPartition
    with pytest.raises(ValueError):
        GPO(partition=partition, algo=algo)


def test_GPO_value_error_2():
    # no partition
    algo = HCT
    domain = [[-5, 5], [-5, 5]]
    with pytest.raises(ValueError):
        GPO(domain=domain, algo=algo)


def test_GPO_value_error_3():
    # no algorithm
    domain = [[-5, 5], [-5, 5]]
    partition = BinaryPartition
    with pytest.raises(ValueError):
        GPO(domain=domain, partition=partition)


def test_GPO_not_implemented_error():
    # wrong algorithm
    T = 100
    Target = Garland.Garland()
    domain = [[0, 1]]
    partition = BinaryPartition
    with pytest.raises(NotImplementedError):
        GPO(rounds=T, domain=domain, partition=partition, algo=StoSOO)


def test_GPO_HOO_Garland():

    T = 1000
    Target = Garland.Garland()
    domain = [[0, 1]]
    partition = BinaryPartition

    algo = GPO(rounds=T, domain=domain, partition=partition, algo=T_HOO)
    GPO_regret_list = []
    GPO_regret = 0

    for t in range(1, T + 1):
        point = algo.pull(t)
        reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
        algo.receive_reward(t, reward)
        inst_regret = Target.fmax - Target.f(point)
        GPO_regret += inst_regret
        GPO_regret_list.append(GPO_regret)
    p = algo.get_last_point()
    inst_regret = Target.fmax - Target.f(p)

    # regret_dic = {"GPO": np.array(GPO_regret_list),}
    # compare_regret(regret_dic)

def test_GPO_HCT_Garland():

    T = 1000
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
    regret_dic = {"GPO": np.array(GPO_regret_list)}
    compare_regret(regret_dic)

test_GPO_HCT_Garland()