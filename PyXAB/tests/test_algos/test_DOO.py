from PyXAB.synthetic_obj import *

from PyXAB.algos.DOO import DOO
from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import compare_regret

import math
import numpy as np
import pdb
import pytest

def test_DOO_ValueError_1():
    partition = BinaryPartition
    with pytest.raises(ValueError):
        algo = DOO(partition=partition)

def test_DOO_ValueError_2():
    domain = [[0, 1]]
    with pytest.raises(ValueError):
        algo = DOO(domain=domain)

def test_DOO_DoubleSine():
    T = 500
    Target = DoubleSine.DoubleSine()
    domain = [[0, 1]]
    partition = BinaryPartition
    algo = DOO(n=T, domain=domain, partition=partition)

    for t in range(1, T + 1):
        point = algo.pull(t)
        reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
        algo.receive_reward(t, reward)
        
    last_point = algo.get_last_point()
    print(T, Target.fmax - Target.f(last_point))

def test_DOO_Ackley():
    T = 500
    Target = Ackley.Ackley_Normalized()
    domain = [[0, 1], [0, 1]]
    partition = BinaryPartition
    algo = DOO(n=T, domain=domain, partition=partition)

    for t in range(1, T + 1):
        point = algo.pull(t)
        reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
        algo.receive_reward(t, reward)
        
    last_point = algo.get_last_point()
    print(T, Target.fmax - Target.f(last_point))
    
def test_DOO_Garland():
    T = 500
    Target = Garland.Garland()
    domain = [[0, 1]]
    partition = BinaryPartition
    algo = DOO(n=T, domain=domain, partition=partition)

    for t in range(1, T + 1):
        point = algo.pull(t)
        reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
        algo.receive_reward(t, reward)
        
    last_point = algo.get_last_point()
    print(T, Target.fmax - Target.f(last_point))