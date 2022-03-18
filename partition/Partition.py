from partition.Node import P_node
import math
import numpy as np
import pdb


### Implementation of baseline partition

class Partition:

    def __init__(self, domain):

        self.domain = domain
        self.root = P_node(0, 1, None, domain)
        self.depth = 0
        self.node_list = [[self.root]]


    def deepen(self):

        self.depth += 1
        self.make_children()


    def make_children(self):

        # Every user-defined partition needs to re-write this function
        # Otherwise error is  thrown

        raise NotImplementedError

    def get_node(self, depth, index):

        if depth > len(self.node_list):
            raise ValueError('Layer not yet constructed')
        else:
            if index-1 > len(self.node_list[depth]):
                raise ValueError('Index Outside of Range')

        return self.node_list[depth][index-1]

    def get_node_list(self):

        return self.node_list


    def get_root(self):

        return self.root

    def get_depth(self):

        return self.depth