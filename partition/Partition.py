from partition.Node import P_node
import math
import numpy as np
import pdb


### Implementation of baseline partition (No partition)

class Partition:

    def __init__(self, domain):

        self.domain = domain
        self.root = P_node(0, 1, None, domain)
        self.height = 0
        self.node_list = [[self.root]]


    def deepen(self):

        self.height += 1
        self.make_children()

    # Every user-defined partition needs to re-write this function

    def make_children(self):

        new_deepest = []
        for node in self.node_list[-1]:

            child = P_node(depth=node.get_depth()+1, index=node.get_index(),
                           parent=node, domain=node.get_domain())
            children = [child]
            node.update_children(children)
            new_deepest = new_deepest + children

        self.node_list.append(new_deepest)

    def get_node(self, depth, index):

        if depth > len(self.node_list):
            raise ValueError('Layer not yet constructed')
        else:
            if index-1 > len(self.node_list[depth]):
                raise ValueError('Index Outside of Range')

        return self.node_list[depth][index-1]

    def get_node_list(self):

        return self.node_list