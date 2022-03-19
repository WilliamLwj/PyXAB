from synthetic_obj.Garland import Garland
from algos.HOO import T_HOO
from partition.BinaryPartition import BinaryPartition
from utils import plot_regret
import numpy as np

T = 5000
Target = Garland()
domain = [[0, 1]]
partition = BinaryPartition(domain)
algo = T_HOO(rounds=T, partition=partition)

cumulative_regret = 0
cumulative_regret_list = [0]

for t in range(T):

    point = algo.pull(t)
    reward = Target.f(point) + np.random.normal(0, 0.1)
    algo.receive_reward(t, reward)
    inst_regret = Target.fmax - Target.f(point)

    cumulative_regret += inst_regret
    print(point)

    # print(cumulative_regret)
    cumulative_regret_list.append(cumulative_regret)

plot_regret(np.array(cumulative_regret_list), 'T-HOO')