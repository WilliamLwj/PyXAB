from PyXAB.partition.Node import P_node
from abc import ABC, abstractmethod
import pdb
### Implementation of baseline partition

class Partition(ABC):

    def __init__(self, domain, node=P_node):

        self.domain = domain
        self.root = node(0, 1, None, domain)
        self.depth = 0
        self.node_list = [[self.root]]


    def deepen(self):

        depth = self.depth
        for i in range(len(self.node_list[depth])):
            parent = self.node_list[depth][i]
            if i == 0:
                self.make_children(parent, newlayer=True)
            else:
                self.make_children(parent, newlayer=False)

    @abstractmethod
    def make_children(self, parent, newlayer=False):

        # Every user-defined partition needs to re-write this function
        # Otherwise error is  thrown

        pass


    def get_layer_node_list(self, depth):

        return self.node_list[depth]

    def get_node_list(self):

        return self.node_list


    def get_root(self):

        return self.root

    def get_depth(self):

        return self.depth