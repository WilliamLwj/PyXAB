# -*- coding: utf-8 -*-
"""Implementation of StroquOOL (Bartlett, 2019)
"""
# Author: Haoze Li <li4456@purdue.edu>
# License: MIT


import math
import numpy as np
from PyXAB.algos.Algo import Algorithm
from PyXAB.partition.Node import P_node
import pdb

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
    
    def is_opened(self):
        return True if self.opened else False
    
    def open_node(self):
        self.opened = True
    
    def remove_reward(self):
        self.rewards = []

class StroquOOL(Algorithm):
    
    def __init__(self, n=100, domain=None, partition=None):
        super(StroquOOL).__init__()
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        self.partition = partition(domain=domain, node=StroquOOL_node)
        # self.iteration = 0
        self.h_max = math.floor(n / (2 * (np.log2(n) + 1)**2))
        self.p_max = math.floor(np.log2(self.h_max))
        
        self.curr_node = self.partition.root
        self.curr_depth = 0
        self.chosen = []
        
    def pull(self, depth):
        
        self.curr_depth = depth
        node_list = self.partition.get_node_list()
        point = []
        
        if self.curr_depth <= self.h_max:
            for p in range(math.floor(np.log2(self.h_max / (self.curr_depth + 1))), -1, -1):
                max_reward = -np.inf
                for i in range(len(node_list[self.curr_depth])):
                    node = node_list[self.curr_depth][i]
                    if not node.is_opened() & node.get_visited_times() >= 2**p:
                        node.compute_mean_reward()
                        if node.get_mean_reward() >= max_reward:
                            max_reward = node.get_mean_reward()
                            max_node = node
                            # point_index = i
                            
                max_node.open_node()
                if max_node.get_children() is None: 
                    self.partition.make_children(max_node, newlayer=True)
                    for child in max_node.get_children():
                        point.append((child, p)) 
                        self.chosen.append(child) # build T_{h, i}
        return point
    
    def next_layer(self):
        
        self.curr_depth += 1
    
    def receive_reward(self, node, reward):
        
        node.visited_times += 1
        node.update_reward(reward)
    
    def get_chosen(self):
        return self.chosen
    
    def get_h_max(self):
        return self.h_max
