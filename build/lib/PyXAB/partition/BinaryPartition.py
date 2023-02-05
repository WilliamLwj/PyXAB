# -*- coding: utf-8 -*-
"""Implmentation of Binary Partition
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT

from PyXAB.partition.Node import P_node
from PyXAB.partition.Partition import Partition
import numpy as np
import pdb

class BinaryPartition(Partition):

    def __init__(self, domain, node=P_node):

        super(BinaryPartition, self).__init__(domain=domain, node=node)

    # Rewrite the make_children function in the Partition class
    def make_children(self, parent, newlayer=False):


        parent_domain = parent.get_domain()
        dim = np.random.randint(0, len(parent_domain))
        selected_dim = parent_domain[dim]

        domain1 = parent_domain.copy()
        domain2 = parent_domain.copy()

        domain1[dim] = [selected_dim[0], (selected_dim[0] + selected_dim[1]) / 2]
        domain2[dim] = [(selected_dim[0] + selected_dim[1]) / 2, selected_dim[1]]

        node1 = self.node(depth=parent.get_depth() + 1, index=2 * parent.get_index() - 1,
                       parent=parent, domain=domain1)
        node2 = self.node(depth=parent.get_depth() + 1, index=2 * parent.get_index(),
                       parent=parent, domain=domain2)
        parent.update_children([node1, node2])


        new_deepest = []
        new_deepest.append(node1)
        new_deepest.append(node2)

        if newlayer:
            self.node_list.append(new_deepest)
            self.depth += 1
        else:
            self.node_list[parent.get_depth() + 1] += new_deepest
