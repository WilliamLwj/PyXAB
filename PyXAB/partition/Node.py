# -*- coding: utf-8 -*-
"""Base Implmentation of a Node
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT


import math
import numpy as np
import pdb


class P_node:
    """
    The most basic node class that contains everything needed to be inside a partition

    """

    def __init__(self, depth, index, parent, domain):
        """
        Initialization of the P_node class

        Parameters
        ----------
        depth: int
            The depth of the node

        index: int
            The index of the node

        parent:
            The parent node of the P_node

        domain: list(list)
            The domain that this P_node represents

        """
        self.depth = depth
        self.index = index
        self.parent = parent
        self.children = None
        self.domain = domain

        point = []
        for x in self.domain:
            point.append((x[0] + x[1]) / 2)

            # TODO: Different Domains Other Than Continuous Domains

        self.c_point = point

    def update_children(self, children):
        """
        The function to update the children of the node

        Parameters
        ----------
        children:
            The children nodes to be updated

        Returns
        -------

        """
        self.children = children

    def get_cpoint(self):
        """
        The function to get the center point of the domain

        Returns
        -------

        """
        return self.c_point

    def get_children(self):
        """
        The function to get the children of the node

        Returns
        -------

        """
        return self.children

    def get_parent(self):
        """
        The function to get the parent of the node

        Returns
        -------

        """
        return self.parent

    def get_domain(self):
        """
        The function to get the domain of the node

        Returns
        -------

        """
        return self.domain

    def get_depth(self):
        """
        The function to get the depth of the node

        Returns
        -------

        """
        return self.depth

    def get_index(self):
        """
        The function to get the index of the node

        Returns
        -------

        """
        return self.index
