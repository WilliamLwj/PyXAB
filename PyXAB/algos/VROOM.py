# -*- coding: utf-8 -*-
"""Implementation of VROOM (Ammar, Haitham, et al., 2020)
"""
# Author: Haoze Li <li4456@purdue.edu>
# License: MIT

import math
import numpy as np
from PyXAB.algos.Algo import Algorithm
from PyXAB.partition.Node import P_node
from PyXAB.partition.BinaryPartition import BinaryPartition
import pdb

class VROOM_node(P_node):
    """
    Implementation of the node in the VROOM algorithm
    """
    
    def __init__(self, depth, index, parent, domain):
        """
        Initialization of the VROOM node
        
        Parameters
        ----------
        depth: int
            depth of the node
        index: int
            index of the node
        parent: 
            parent node of the current node
        domain: list(list)
            domain that this node represents
        """
        super(VROOM_node, self).__init__(depth, index, parent, domain)
        
        self.reward = []
        self.rank = []
        self.reward_tilde = []
        
    def update_reward(self, reward):
        """
        The function to update the reward of the node
        
        Parameters
        ----------
        reward: float
            the reward for evaluating the node
            
        Returns
        -------
        
        """
        self.reward.append(reward)

    def update_reward_tilde(self, reward):
        """
        The function to update the reward tilde of the node
        
        Parameters
        ----------
        reward: float
            the reward tilde statistc of the node
        
        Returns
        -------
        """
        self.reward_tilde.append(reward)

    def get_mean_reward(self):
        """
        The function to get the mean of the reward of the node
        
        Returns
        -------
        
        """
        return np.mean(self.reward)
    
    def get_reward_tilde(self):
        """
        The function to get the reward tilde statistic of the node
        
        Returns
        -------
    
        """
        if self.reward_tilde:
            return np.sum(self.reward)
        return -np.inf
        
    def get_eval_time(self):
        """
        The function to get the evaluation time of the node
        
        Returns
        -------
        
        """
        return len(self.reward)
        
    def sample_uniform(self):
        """
        The function to uniformly sample a point from the domain of the node
        
        Returns
        -------
        res: list
            the point sampled by the sampler
        """
        # TODO: extend the function to the case where the domain is of the form [a, b]\cup [c, d]
        res = []
        for domain in self.domain:
            point = np.random.uniform(domain[0], domain[1])
            res.append(point)
        return res
    
    def get_rank(self):
        """
        The function to get the rank of the cell
        
        Returns
        -------
        rank: int
            the rank of the cell at current depth
        """
        return self.rank
    
    def add_rank(self, rank):
        """
        The method to set the rank of the cell
        
        Parameters
        ----------
        rank: int
            the rank of the cell at current depth
        """
        self.rank.append(rank)
    
    
class VROOM(Algorithm):
    """
    The implementation of the VROOM algorithm (Ammar, Haitham, et al., 2020)
    """
    
    def __init__(self, n=100, h_max = 100, b=None, f_max=None, domain=None, partition=BinaryPartition):
        """
        The initialization of the VROOM algorithm
        
        Parameters
        ----------
        n: int
            The total number of rounds (budget)
        b: float
            The parameter that measures the variation of the function
        f_max: float
            An upper bound of the objective function
        domain: list(list)
            The domain of the objective to be optimized
        partition:
            The partition choice of the algorithm
        """
        super(VROOM, self).__init__()
        if b is None:
            raise ValueError("Variance parameter is not given.")
        if f_max is None:
            raise ValueError("Upper bound of the objective function is not given.")
        if domain is None:
            raise ValueError("Parameter space is not given.")
        if partition is None:
            raise ValueError("Partition of the parameter space is not given")
        self.partition = partition(domain=domain, node=VROOM_node)
        
        self.iteration = 0
        self.n = n
        self.b = b
        self.f_max = f_max
        self.search_depth = math.floor(np.log2(n)) # the largest depth at the ranking stage
        self.delta = 4 * self.b / (self.f_max * np.sqrt(self.n))
        self.h_max = h_max
        
        # generate the searching tree
        while self.partition.get_depth() < self.search_depth:
            self.partition.deepen()
            
                
        # calculate the constant 
        self.const = 0
        for h in range(1, self.search_depth + 1):
            for l in range(1, 2**h + 1):
                self.const += 1 / (h * l)
        
    def pull(self, time):
        """
        The pull function of VROOM that returns a point in every bound
        
        Parameters
        ----------
        time: int
            time stamp parameter
            
        Returns
        -------
        point: list
            the point to be evaluated
        """
        
        self.iteration = time
        node_list = self.partition.get_node_list()

        # sample node
        index = []
        self.prob = []
        for h in range(1, self.search_depth + 1):
            self.rank(node_list[h])
            for l in range(len(node_list[h])):
                index.append((h, l))
                # print(node_list[h][l].get_rank()[-1])
                self.prob.append(1 / (h * node_list[h][l].get_rank()[-1] * self.const))
        sample = np.random.choice([i for i in range(len(index))], p=self.prob)
        idx = index[sample]
        self.curr_node = node_list[idx[0]][idx[1]]
        node = node_list[idx[0]][idx[1]]
        
        # sample point
        h = idx[0]
        self.update_list = [node]
        while h < self.h_max:
            if node.get_children() is None:
                self.partition.make_children(node, newlayer=(h >= self.partition.get_depth()))
            sign = np.random.randint(2)
            node = node.get_children()[sign]
            self.update_list.append(node)
            h += 1
        return node.sample_uniform()
    
    def rank(self, nodes):
        """
        The rank function of VROOM that rank nodes at the same depth
        
        Parameters
        ----------
        nodes: list
            a list of node at the same depth
            
        Returns
        -------
        """
        def rank_fun(node):
            if node.get_eval_time() == 0:
                return -np.inf
            return node.get_mean_reward() - np.sqrt(np.log(4 * self.n**3 / self.delta) / (2 * node.get_eval_time()))
        rank = sorted(nodes, key=rank_fun, reverse=True)
        for i in range(len(rank)):
            node = rank[i]
            node.add_rank(i + 1)
        
    def receive_reward(self, time, reward):
        """
        The receive_reward function of VROOM to obtain the reward and update Statistics (for current node)
        
        Parameters
        ----------
        time: int
            The time stamp parameter
        reward: float
            The reward of the evaluation
        
        Returns
        -------
        """
        
        for i in range(len(self.update_list)):
            node = self.update_list[i]
            depth = node.get_depth()
            prob = 0
            idx = 0
            for h in range(1, depth + 1):
                for l in range(1, 2**h + 1):
                    prob += self.prob[idx]
                    idx += 1
                if idx >= len(self.prob):
                    prob = 1
                    break
            node.update_reward(reward)
            node.update_reward_tilde(reward / (prob / (2**i))) # Eq. (4) in the paper
        
    def get_last_point(self):
        """
        The function to get the last point in VROOM
        
        Returns
        -------
        point: list
            The output of the VROOM algorithm at last
        """
        max_value = -np.inf
        max_node = None
        node_list = self.partition.get_node_list()
        for h in range(len(node_list)):
            for node in node_list[h]:
                value = node.get_reward_tilde() - self.f_max * np.sqrt(2 * self.n * self.const * np.log(2 * self.n **2 / self.delta) * np.sum(node.get_rank())) + self.f_max * self.const * (np.log(2 * self.n **2 / self.delta) / 3)
                if value >= max_value:
                    max_node = node
                    depth = h
                    max_value = value
        while depth < self.h_max:
            if max_node.get_children() is None:
                self.partition.make_children(max_node, newlayer=(depth >= self.partition.get_depth()))
            sign = np.random.randint(2)
            max_node = max_node.get_children()[sign]
            depth += 1
        return max_node.sample_uniform()