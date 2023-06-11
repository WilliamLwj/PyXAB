from PyXAB.synthetic_obj import *

from PyXAB.algos.VROOM import VROOM
from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import compare_regret

import math
import numpy as np
import pdb
import pytest


def test_VROOM_ValueError_1():
    partition = BinaryPartition
    b = 1
    f_max = 10
    with pytest.raises(ValueError):
        VROOM(b=b, f_max = f_max, partition=partition)


def test_VROOM_ValueError_2():
    domain = [[0, 1]]
    b = 1
    f_max = 10
    with pytest.raises(ValueError):
        VROOM(b=b, f_max = f_max, domain=domain, partition=None)

def test_VROOM_ValueError_3():
    partition = BinaryPartition
    domain = [[0, 1]]
    f_max = 10
    with pytest.raises(ValueError):
        VROOM(f_max = f_max, domain=domain, partition=partition)
        
def test_VROOM_ValueError_4():
    partition = BinaryPartition
    domain = [[0, 1]]
    b = 1
    with pytest.raises(ValueError):
        VROOM(b = b, domain=domain, partition=partition)

def test_VROOM_Garland():
    T = 100
    Target = Garland.Garland()
    domain = [[0, 1]]
    partition = BinaryPartition
    f_max = 1
    b = 0.5
    algo = VROOM(n=T, f_max = f_max, b = b, domain=domain, partition=partition)

    for t in range(1, T + 1):
        point = algo.pull(t)
        reward = Target.f(point)
        algo.receive_reward(t, reward)

    last_point = algo.get_last_point()
    print(T, Target.fmax - Target.f(last_point), last_point)
    
def test_VROOM_DoubleSine():
    T = 100
    Target = DoubleSine.DoubleSine()
    domain = [[0, 1]]
    partition = BinaryPartition
    f_max = 1
    b = 0.5
    algo = VROOM(n=T, f_max = f_max, b = b, domain=domain, partition=partition)

    for t in range(1, T + 1):
        point = algo.pull(t)
        reward = Target.f(point)
        algo.receive_reward(t, reward)

    last_point = algo.get_last_point()
    print(T, Target.fmax - Target.f(last_point), last_point)


def test_SOO_Ackley():
    T = 100
    Target = Ackley.Ackley_Normalized()
    domain = [[0, 1], [0, 1]]
    partition = BinaryPartition
    f_max = 5
    b = 1
    algo = VROOM(n=T, f_max = f_max, b = b, domain=domain, partition=partition)

    for t in range(1, T + 1):
        point = algo.pull(t)
        reward = Target.f(point)
        algo.receive_reward(t, reward)

    last_point = algo.get_last_point()
    print(T, Target.fmax - Target.f(last_point))
    # plot the regret
    # regret_dic = {"POO": np.array(POO_regret_list)}
    # compare_regret(regret_dic)


def test_VROOM_SmallSearchingDepth():
    T = 100
    Target = Garland.Garland()
    domain = [[0, 1]]
    partition = BinaryPartition
    f_max = 5
    b = 1
    algo = VROOM(n=T, f_max = f_max, b = b, domain=domain, partition=partition)

    for t in range(1, T + 1):
        point = algo.pull(t)
        reward = Target.f(point)
        algo.receive_reward(t, reward)

    last_point = algo.get_last_point()
    print(T, Target.fmax - Target.f(last_point))
    