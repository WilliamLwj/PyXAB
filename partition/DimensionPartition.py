from partition.Node import P_node
from partition.Partition import Partition
import math
import numpy as np
import pdb




class DimensionBinaryPartition(Partition):

    def __init__(self, domain):

        super(DimensionBinaryPartition, self).__init__(domain)

    # Rewrite the make_children function in the Partition class
    def make_children(self):

        new_deepest = []

        for node in self.node_list[-1]:
            parent_domain = node.get_domain()
            children_list = []
            num_children = 2 ** len(parent_domain)

            combination_list = []
            for dim in range(len(parent_domain)):
                selected_dim = parent_domain[dim]
                split_point = (selected_dim[0] + selected_dim[1]) / 2
                comb1= [selected_dim[0], split_point]
                comb2 = [split_point, selected_dim[1]]

                combination_list.append([comb1, comb2])

            for i in range((num_children)):


                ind = i
                domain = []
                for dim in range(len(parent_domain)):
                    j = ind // 2**(len(parent_domain) - dim - 1)
                    domain.append(combination_list[len(parent_domain) - dim - 1][j])
                    ind = ind - j * 2**(len(parent_domain) - dim - 1)

                domain.reverse()
                new_node = P_node(depth=node.get_depth() + 1, index= num_children * (node.get_index()-1) + i + 1,
                               parent=node, domain=domain)

                children_list.append(new_node)

            node.update_children(children_list)
            new_deepest += children_list

        self.node_list.append(new_deepest)

