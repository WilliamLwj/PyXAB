from synthetic_obj.Garland import Garland
from algos.HOO import T_HOO, HOO_tree
from partition.BinaryPartition import BinaryPartition
from utils import plot_regret, compare_regret
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
    reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
    algo.receive_reward(t, reward)
    inst_regret = Target.fmax - Target.f(point)

    cumulative_regret += inst_regret
    print(point)

    # print(cumulative_regret)
    cumulative_regret_list.append(cumulative_regret)

# plot_regret(np.array(cumulative_regret_list), 'T-HOO')

HOO_regret_list = []
regret = 0
print("HOO training")
tree = HOO_tree(1, 0.75, domain, T)
for i in range(T):
    curr_node, path = tree.optTraverse()
    sample_range = curr_node.range
    pulled_x = []
    for j in range(len(sample_range)):
        x = (sample_range[j][0] + sample_range[j][1]) / 2.0
        pulled_x.append(x)
    reward = Target.f(pulled_x) + np.random.uniform(-0.1, 0.1)
    tree.updateAllTree(path, reward)

    simple_regret = Target.fmax - Target.f(pulled_x)
    regret += simple_regret
    HOO_regret_list.append(regret)
            # print(i, pulled_x)

regret_dic = {'T-HOO': np.array(cumulative_regret_list),
              'HOO': np.array(HOO_regret_list)}
compare_regret(regret_dic)