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
    """
    Implementation of Binary Partition
    """

    def __init__(self, domain, node=P_node):
        """
        Initialization of the Binary Partition

        Parameters
        ----------
        domain: list(list)
            The domain of the objective function to be optimized, should be in the form of list of lists (hypercubes),
            i.e., [[range1], [range2], ... [range_d]], where [range_i] is a list indicating the domain's projection on
            the i-th dimension, e.g., [-1, 1]

        node
            The node used in the partition, with the default choice to be P_node.
        """
        super(BinaryPartition, self).__init__(domain=domain, node=node)

    # Rewrite the make_children function in the Partition class
    def make_children(self, parent, newlayer=False):
        """
        The function to make children for the parent node with a standard binary partition, i.e., split every
        parent node in the middle. If there are multiple dimensions, the dimension to split the parent is chosen
        randomly

        Parameters
        ----------
        parent:
            The parent node to be expanded into children nodes

        newlayer: bool
            Boolean variable that indicates whether or not a new layer is created

        Returns
        -------

        """

        parent_domain = parent.get_domain()
        dim = np.random.randint(0, len(parent_domain))
        selected_dim = parent_domain[dim]

        domain1 = parent_domain.copy()
        domain2 = parent_domain.copy()

        domain1[dim] = [selected_dim[0], (selected_dim[0] + selected_dim[1]) / 2]
        domain2[dim] = [(selected_dim[0] + selected_dim[1]) / 2, selected_dim[1]]

        node1 = self.node(
            depth=parent.get_depth() + 1,
            index=2 * parent.get_index() - 1,
            parent=parent,
            domain=domain1,
        )
        node2 = self.node(
            depth=parent.get_depth() + 1,
            index=2 * parent.get_index(),
            parent=parent,
            domain=domain2,
        )
        parent.update_children([node1, node2])

        new_deepest = []
        new_deepest.append(node1)
        new_deepest.append(node2)

        if newlayer:
            self.node_list.append(new_deepest)
            self.depth += 1
        else:
            self.node_list[parent.get_depth() + 1] += new_deepest
