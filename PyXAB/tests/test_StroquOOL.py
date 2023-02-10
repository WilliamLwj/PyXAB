from PyXAB.synthetic_obj import *

# from PyXAB.algos.StroquOOL import StroquOOL
from PyXAB.partition.BinaryPartition import BinaryPartition
from PyXAB.utils.plot import compare_regret
import math
import numpy as np
import pdb


# -*- coding: utf-8 -*-
"""Implementation of StroquOOL (Bartlett, 2019)
"""
# Author: Haoze Li <li4456@purdue.edu>
# License: MIT


# import math
# import numpy as np
from PyXAB.algos.Algo import Algorithm
from PyXAB.partition.Node import P_node
# import pdb

class StroquOOL_node(P_node):
    
    def __init__(self, depth, index, parent, domain):
        super(StroquOOL_node, self).__init__(depth, index, parent, domain)
        
        self.visited_times = 0  # store the number of evaluations of the node
        self.opened = False
        self.rewards = []  # list of rewards after obtaining the evaluation of some point in the domain of this node
        self.mean_reward = 0
    
    def update_reward(self, reward):
        
        self.rewards.append(reward)
    
    def get_visited_times(self):
        
        return self.visited_times
    
    def compute_mean_reward(self):
        
        if self.visited_times == 0:
            self.mean_reward = -np.inf
        else:
            self.mean_reward = np.sum(np.array(self.rewards)) / len(self.rewards)

    def get_mean_reward(self):
        
        return self.mean_reward
    
    def not_opened(self):
        return False if self.opened else True
    
    def open_node(self):
        self.opened = True
    
    def remove_reward(self):
        self.rewards = []
        
    def get_randompoint(self):
        
        randompoint = []
        for x in self.domain:
            # Randomly chosen point from cotinuous domain
            randompoint.append(np.random.uniform(x[0], x[1]))
        return randompoint
    
    
class StroquOOL(Algorithm):
    
    def __init__(self, n=1000, domain=None, partition=None):
        super(StroquOOL).__init__()
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        self.partition = partition(domain=domain, node=StroquOOL_node)
        self.iteration = 0
        self.h_max = math.floor(n / (2 * (self.harmonic_series_sum(n) + 1)**2))
        self.p_max = math.floor(np.log2(self.h_max))
        
        self.curr_depth = 0
        self.curr_p = math.floor(np.log2(self.h_max / (self.curr_depth + 1)))
        self.chosen = []
        self.time_stamp = 0 # store the time point at which the algorithm finishes one open procedure
        self.validation_p = 0
        self.candidate = []
        self.curr_loc = 0
        self.curr_node = self.partition.root
        self.eval = True
        self.max_node = None
    
    @staticmethod
    def harmonic_series_sum(n):
        
        res = 0
        for i in range(1, n + 1):
            res += 1 / i
        return res
        
    def reset_p(self):
        
        self.curr_p = math.floor(np.log2(self.h_max / (self.curr_depth + 1)))
    
    # def next_child(self):
        
    #     if self.flag_which_child == 0:
    #         self.flag_which_child = 1
    #     else:
    #         self.flag_which_child = 0
        
    def pull(self, time):
        
        self.iteration = time
        node_list = self.partition.get_node_list()
        
        if self.curr_depth < self.h_max:
            flag = True
            # init
            if self.curr_depth == 0:
                if node_list[0][0].get_children() is None: 
                    self.partition.make_children(node_list[0][0], newlayer=True)
                    self.chosen.append(node_list[0][0].get_children()[0])
                    self.chosen.append(node_list[0][0].get_children()[1])
                if self.iteration <= self.h_max:
                    self.curr_node = node_list[0][0].get_children()[0]
                    return node_list[0][0].get_children()[0].get_cpoint()
                if self.h_max < self.iteration <= 2 * self.h_max: 
                    if self.iteration == 2*self.h_max:
                        self.time_stamp = 2 * self.h_max
                        self.curr_depth += 1
                        self.reset_p()
                    self.curr_node = node_list[0][0].get_children()[1]
                    return node_list[0][0].get_children()[1].get_cpoint()
                    
            if self.curr_p >= 0:        
                # Choose node to open
                if self.eval:
                    max_reward = -np.inf
                    for i in range(len(node_list[self.curr_depth])):
                        node = node_list[self.curr_depth][i]
                        print(i, self.curr_depth)
                        if node.not_opened() and node.get_visited_times() >= 2**self.curr_p:
                            node.compute_mean_reward()
                            if node.get_mean_reward() >= max_reward:
                                max_reward = node.get_mean_reward()
                                self.max_node = node
                    self.eval = False
                    # partition                   
                    if self.max_node.get_children() is None:
                        self.partition.make_children(self.max_node, newlayer=True)
                        self.chosen.append(self.max_node.get_children()[0])
                        self.chosen.append(self.max_node.get_children()[1])
                # evaluate children
                if self.iteration <= self.time_stamp + 2**self.curr_p:
                    self.curr_node = self.max_node.get_children()[0]
                    return self.max_node.get_children()[0].get_cpoint()
                if self.time_stamp + 2**self.curr_p < self.iteration <= self.time_stamp + 2**(self.curr_p + 1):
                    if self.iteration == self.time_stamp + 2**(self.curr_p + 1):
                        self.time_stamp += 2**(self.curr_p + 1) 
                        self.curr_p -= 1
                        self.max_node.open_node()
                        self.eval = True
                        if self.curr_p < 0:
                            self.curr_depth += 1
                            self.reset_p()
                    self.curr_node = self.max_node.get_children()[1]
                    return self.max_node.get_children()[1].get_cpoint()  
        # Cross-Validation
        else:
            # get candidate set
            if not self.candidate:
                for p in range(self.p_max + 1):
                    max_value = -np.inf
                    max_node = None
                    for i in range(len(self.chosen)):
                        if self.chosen[i].get_visited_times() >= 2**p:
                            if self.chosen[i].get_mean_reward() >= max_value:
                                max_value = self.chosen[i].get_mean_reward()
                                max_node = self.chosen[i]
                    self.candidate.append(max_node)   
                for node in self.candidate:
                    node.remove_reward()
                          
            if self.curr_loc < len(self.candidate):
                if self.iteration <= self.time_stamp + self.h_max:
                    self.curr_node = self.candidate[self.curr_loc]
                    if self.iteration == self.time_stamp + self.h_max:
                        self.time_stamp += self.h_max
                        self.curr_loc += 1
                    return self.curr_node.get_cpoint()

                    
                
            # if self.validation_p <= self.p_max:
            #     max_value = -np.inf
            #     max_node = None 
            #     for i in range(len(self.chosen)):
            #         if self.chosen[i].get_visited_times() >= 2**self.validation_p:
            #             if self.chosen[i].get_mean_reward() >= max_value:
            #                 max_value = self.chosen[i].get_mean_reward()
            #                 max_node = self.chosen[i]
                            
            #     if self.iteration <= self.time_stamp + self.h_max:
            #         if self.iteration == self.time_stamp + self.h_max:
            #             self.time_stamp += self.h_max
            #             self.validation_p += 1
            #             self.candidate.append(max_node)
            #         self.curr_node = max_node
            #         return max_node.get_cpoint()

                    
                # max_value = -np.inf
                # max_node = None
                # for i in self.position:
                #     if chosen[i].get_mean_reward() >= max_value:
                #         max_node = chosen[i]
                # self.last_point = max_node.get_cpoint()            
                
                    
                        
                
        
        
        
        
        # self.curr_depth = depth
        # node_list = self.partition.get_node_list()
        # point = []
        
        # if self.curr_depth <= self.h_max:
        #     for p in range(math.floor(np.log2(self.h_max / (self.curr_depth + 1))), -1, -1):
        #         max_reward = -np.inf
        #         max_node = None
        #         for i in range(len(node_list[self.curr_depth])):
        #             node = node_list[self.curr_depth][i]
        #             # print(node.get_visited_times(), p)
        #             if node.not_opened() and node.get_visited_times() >= 2**p:
        #                 node.compute_mean_reward()
        #                 if node.get_mean_reward() >= max_reward:
        #                     max_reward = node.get_mean_reward()
        #                     max_node = node
        #                     # point_index = i
                            
        #         if max_node is not None:            
        #             max_node.open_node()
        #             if max_node.get_children() is None: 
        #                 self.partition.make_children(max_node, newlayer=True)
        #                 for child in max_node.get_children():
        #                     point.append((child, p)) 
        #                     self.chosen.append(child) # build T_{h, i}
        # return point
    
    def next_layer(self):
        
        self.curr_depth += 1
    
    def receive_reward(self, t, reward):
        
        
        self.curr_node.visited_times += 1
        self.curr_node.update_reward(reward)
    
    def get_chosen(self):
        return self.chosen
    
    def get_h_max(self):
        return self.h_max

    def get_last_point(self):
        
        max_value = -np.inf
        max_node = None
        for node in self.candidate:
            node.compute_mean_reward()
            if node.get_mean_reward() >= max_value:
                max_value = node.get_mean_reward()
                max_node = node
        return max_node.get_cpoint()

T = 20000
H = math.floor(T / (2 * (np.log2(T) + 1)**2))
# Target = Garland.Garland()
Target = DoubleSine.DoubleSine()
domain = [[0, 1]]
partition = BinaryPartition
algo = StroquOOL(n = T, domain=domain, partition=partition)

for t in range(1, T + 1):
    point = algo.pull(t)
    print(point)
    if point is None: 
        break
    reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
    algo.receive_reward(t, reward)

last_point = algo.get_last_point()
print(algo.iteration, Target.fmax - Target.f(last_point), last_point)

# for j in range(len(algo.partition.root.get_children())):
#     node = algo.partition.root.get_children()[j]
#     for h in range(H):
#         reward = Target.f(node.get_randompoint()) + np.random.uniform(-0.1, 0.1)
#         algo.receive_reward(node, reward)


# for h in range(1, H):
#     # print(h)
#     node_list = algo.pull(h) # nodes at depth = h to open
#     for i in range(len(node_list)):
#         node = node_list[i][0]
#         open_time = node_list[i][1]
#         # index = point_list[i][2]
#         for p in range(2**open_time):
#             reward = Target.f(node.get_randompoint()) + np.random.uniform(-0.1, 0.1)
#             algo.receive_reward(node, reward)
    
    
# # Cross Validation

# chosen = algo.get_chosen()
# p_max = algo.p_max
# max_point = None
# max_value = -np.inf

# for p in range(p_max + 1):
#     for i in range(len(chosen)):
#         # print(chosen[i].get_visited_times())
#         node = chosen[i]
#         if node.get_visited_times() >= 2**p:
#             node.remove_reward()
#             point = node.get_randompoint()
#             for h in range(H):
#                 reward = Target.f(point) + np.random.uniform(-0.1, 0.1)
#                 node.update_reward(reward)
        
#             node.compute_mean_reward()
#             if node.get_mean_reward() >= max_value:
#                 max_value = node.get_mean_reward()
#                 max_point = point

# print(Target.fmax - Target.f(max_point), max_point)

