# -*- coding: utf-8 -*-
"""Base Implmentation of a Partition
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT

from PyXAB.partition.Node import P_node
from abc import ABC, abstractmethod
import pdb


class Partition(ABC):
    """
    Abstract class for partition of the parameter domain
    """

    def __init__(self, domain, node=P_node):
        """
        Initialization of the partition

        Parameters
        ----------
        domain : list(list)
            The domain of the objective function to be optimized, should be in the form of list of lists (hypercubes),
            i.e., [[range1], [range2], ... [range_d]], where [range_i] is a list indicating the domain's projection on
            the i-th dimension, e.g., [-1, 1]

        node
            The node used in the partition, with the default choice to be P_node.

        """
        self.domain = domain
        self.root = node(0, 1, None, domain)
        self.depth = 0
        self.node = node
        self.node_list = [[self.root]]

    def deepen(self):
        """
        The function to deepen the partition by one layer by making children to every node in the last layer

        Returns
        -------

        """
        depth = self.depth
        for i in range(len(self.node_list[depth])):
            parent = self.node_list[depth][i]
            if i == 0:
                self.make_children(parent, newlayer=True)
            else:
                self.make_children(parent, newlayer=False)

    @abstractmethod
    def make_children(self, parent, newlayer=False):
        """
        The function to make children for the parent node

        Parameters
        ----------
        parent:
            The parent node to be expanded into children nodes

        newlayer: bool
            Boolean variable that indicates whether or not a new layer is created

        Returns
        -------

        """

        pass

    def get_layer_node_list(self, depth):
        """
        The function to get the all the nodes on the specified depth

        Parameters
        ----------
        depth: int
            The depth of the layer in the partition

        Returns
        -------
        self.node_list: list
            The list of nodes on the specified depth

        """

        return self.node_list[depth]

    def get_node_list(self):
        """
        The function to get the list all nodes in the partition

        Returns
        -------
        self.node_list: list
            The list of all nodes

        """
        return self.node_list

    def get_root(self):
        """
        The function to get the root of the partition

        Returns
        -------
        self.root:
            The root node of the partition

        """

        return self.root

    def get_depth(self):
        """
        The function to get the depth of the partition

        Returns
        -------
        depth: int
            The depth of the partition
        """

        return self.depth
