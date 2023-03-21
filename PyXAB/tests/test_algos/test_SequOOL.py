from PyXAB.synthetic_obj import *

from PyXAB.algos.SequOOL import SequOOL
from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import compare_regret
import math
import numpy as np
import pdb
import pytest

def test_SequOOL_ValueError_1():
    partition = BinaryPartition
    with pytest.raises(ValueError):
        SequOOL(partition=partition)

def test_SequOOL_ValueError_2():
    domain = [[0, 1]]
    with pytest.raises(ValueError):
        SequOOL(domain=domain)

def test_SequOOL_DoubleSine():
    T = 500
    Target = DoubleSine.DoubleSine()
    domain = [[0, 1]]
    partition = BinaryPartition
    algo = SequOOL(n=T, domain=domain, partition=partition)

    for t in range(1, T + 1):
        point = algo.pull(t)
        reward = Target.f(point)
        algo.receive_reward(t, reward)
        
    last_point = algo.get_last_point()
    print(T, Target.fmax - Target.f(last_point))

def test_SequOOL_Ackley():
    T = 500
    Target = Ackley.Ackley_Normalized()
    domain = [[0, 1], [0, 1]]
    partition = BinaryPartition
    algo = SequOOL(n=T, domain=domain, partition=partition)

    for t in range(1, T + 1):
        point = algo.pull(t)
        reward = Target.f(point)
        algo.receive_reward(t, reward)
        
    last_point = algo.get_last_point()
    print(T, Target.fmax - Target.f(last_point))
    
def test_SequOOL_Garland():
    T = 500
    Target = Garland.Garland()
    domain = [[0, 1]]
    partition = BinaryPartition
    algo = SequOOL(n=T, domain=domain, partition=partition)

    for t in range(1, T + 1):
        point = algo.pull(t)
        reward = Target.f(point)
        algo.receive_reward(t, reward)
        
    last_point = algo.get_last_point()
    print(T, Target.fmax - Target.f(last_point))
        