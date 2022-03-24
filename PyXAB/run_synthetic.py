from PyXAB.synthetic_obj import *
from PyXAB.algos import *

from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import compare_regret
import numpy as np


def main(algo_list, target, domain, partition, noise=0.1, rounds=1000):
    algo_dictionary = {'T-HOO': HOO.T_HOO(rounds=rounds, domain=domain, partition=partition),
                       'HCT': HCT.HCT(domain=domain, partition=partition),
                       'VHCT': VHCT.VHCT(domain=domain, partition=partition),
                       'POO': POO.POO(domain=domain, partition=partition, algo=HOO.T_HOO),
                       'PCT': PCT.PCT(domain=domain, partition=partition)}

    results_dictionary = {}
    for name in algo_list:
        print(name, ": training")
        algo = algo_dictionary[name]
        regret_list = []
        regret = 0
        for t in range(1, rounds + 1):

            print(t)
            point = algo.pull(t)
            reward = target.f(point) + np.random.uniform(-noise, noise)
            algo.receive_reward(t, reward)
            inst_regret = target.fmax - target.f(point)
            regret += inst_regret
            regret_list.append(regret)

        results_dictionary[name] = np.array(regret_list)

    return results_dictionary


target = Garland.Garland()
domain = [[0, 1]]
partition = BinaryPartition
rounds = 1000
noise = 0.1

algo_list = ['T-HOO', 'POO', 'HCT', 'PCT',  'VHCT', ]
regret_dic = main(algo_list, target, domain, partition, noise, rounds)
compare_regret(regret_dic)