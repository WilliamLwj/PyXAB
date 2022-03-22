import math
import numpy as np
import pdb



### Implementation of every node in the partition

class P_node:

    def __init__(self, depth, index, parent, domain):
        self.depth = depth
        self.index = index
        self.parent = parent
        self.children = None
        self.domain = domain

        point = []
        for x in self.domain:

            point.append((x[0] + x[1]) / 2)# TODO: Different Domains Other Than Continuous Domains

        self.c_point = point

    def update_children(self, children):

        self.children = children

    def get_cpoint(self):

        return self.c_point

    def get_children(self):

        return self.children

    def get_parent(self):

        return self.parent

    def get_domain(self):

        return self.domain

    def get_depth(self):

        return self.depth

    def get_index(self):

        return self.index