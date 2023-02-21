from PyXAB.synthetic_obj import *

from PyXAB.algos.StroquOOL import StroquOOL
from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import compare_regret
import math
import numpy as np
import pdb


T = 500
# H = math.floor(T / (2 * (np.log2(T) + 1)**2))
# Target = Garland.Garland()
Target = DoubleSine.DoubleSine()
domain = [[0, 1]]
partition = BinaryPartition
algo = StroquOOL(n=T, domain=domain, partition=partition)

for t in range(1, T + 1):
    point = algo.pull(t)
    reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
    algo.receive_reward(t, reward)

last_point = algo.get_last_point()
print(algo.iteration, Target.fmax - Target.f(last_point), last_point)
