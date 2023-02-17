from PyXAB.synthetic_obj import *

from PyXAB.algos.SequOOL import SequOOL
from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import compare_regret
import math
import numpy as np
import pdb

T = 100
Target = Garland.Garland()
domain = [[0, 1]]
partition = BinaryPartition
algo = SequOOL(n=T, domain=domain, partition=partition)

for t in range(1, T + 1):
    point = algo.pull(t)
    reward = Target.f(point)
    algo.receive_reward(t, reward)

last_point = algo.get_last_point()
print(algo.iteration, Target.fmax - Target.f(last_point), last_point)
