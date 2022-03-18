from partition.Node import P_node
from partition.Partition import Partition
import math
import numpy as np
import pdb


# TODO: Implement the non-standard binary partition

class BinaryPartition(Partition):

    def __init__(self, domain):

        super(BinaryPartition, self).__init__(domain)

    # Rewrite the make_children function in the Partition class
    def make_children(self):

        new_deepest = []

        for node in self.node_list[-1]:
            parent_domain = node.get_domain()
            dim = np.random.randint(0, len(parent_domain))
            selected_dim = parent_domain[dim]

            domain1 = parent_domain.copy()
            domain2 = parent_domain.copy()

            domain1[dim] = [selected_dim[0], (selected_dim[0] + selected_dim[1]) / 2]
            domain2[dim] = [(selected_dim[0] + selected_dim[1]) / 2, selected_dim[1]]

            node1 = P_node(depth=node.get_depth() + 1, index=2 * node.get_index()-1,
                           parent=node, domain=domain1)
            node2 = P_node(depth=node.get_depth() + 1, index=2 * node.get_index(),
                           parent=node, domain=domain2)
            node.update_children([node1, node2])

            new_deepest.append(node1)
            new_deepest.append(node2)

        self.node_list.append(new_deepest)


class RandomBinaryPartition(Partition):

    def __init__(self, domain):

        super(RandomBinaryPartition, self).__init__(domain)

    # Rewrite the make_children function in the Partition class
    def make_children(self):

        new_deepest = []

        for node in self.node_list[-1]:
            parent_domain = node.get_domain()
            dim = np.random.randint(0, len(parent_domain))
            selected_dim = parent_domain[dim]

            domain1 = parent_domain.copy()
            domain2 = parent_domain.copy()

            split_point = np.random.uniform(selected_dim[0], selected_dim[1])
            domain1[dim] = [selected_dim[0], split_point]
            domain2[dim] = [split_point, selected_dim[1]]

            node1 = P_node(depth=node.get_depth() + 1, index=2 * node.get_index()-1,
                           parent=node, domain=domain1)
            node2 = P_node(depth=node.get_depth() + 1, index=2 * node.get_index(),
                           parent=node, domain=domain2)
            node.update_children([node1, node2])

            new_deepest.append(node1)
            new_deepest.append(node2)

        self.node_list.append(new_deepest)