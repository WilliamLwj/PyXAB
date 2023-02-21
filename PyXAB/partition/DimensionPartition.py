"""Implmentation of Dimension-wise Binary Partition
"""

from PyXAB.partition.Node import P_node
from PyXAB.partition.Partition import Partition


class DimensionBinaryPartition(Partition):
    """
    Implementation of Dimension-wise Binary Partition
    """

    def __init__(self, domain, node=P_node):
        """
        Initialization of the Dimension-wise Binary Partition

        Parameters
        ----------
        domain : list(list)
            The domain of the objective function to be optimized, should be in the form of list of lists (hypercubes),
            i.e., [[range1], [range2], ... [range_d]], where [range_i] is a list indicating the domain's projection on
            the i-th dimension, e.g., [-1, 1]

        node
            The node used in the partition, with the default choice to be P_node.

        """

        super(DimensionBinaryPartition, self).__init__(domain=domain, node=node)

    # Rewrite the make_children function in the Partition class
    def make_children(self, parent, newlayer=False):
        """
        The function to make children for the parent node with a dimension-wise binary partition, i.e., split every
        parent node into 2^d children where d is the dimension of the parameter domain. Each dimension of the domain is
        split right in the middle.

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
        children_list = []
        num_children = 2 ** len(parent_domain)

        combination_list = []
        for dim in range(len(parent_domain)):
            selected_dim = parent_domain[dim]
            split_point = (selected_dim[0] + selected_dim[1]) / 2
            comb1 = [selected_dim[0], split_point]
            comb2 = [split_point, selected_dim[1]]

            combination_list.append([comb1, comb2])

        for i in range((num_children)):
            ind = i
            domain = []
            for dim in range(len(parent_domain)):
                j = ind // 2 ** (len(parent_domain) - dim - 1)
                domain.append(combination_list[len(parent_domain) - dim - 1][j])
                ind = ind - j * 2 ** (len(parent_domain) - dim - 1)

            domain.reverse()
            new_node = self.node(
                depth=parent.get_depth() + 1,
                index=num_children * (parent.get_index() - 1) + i + 1,
                parent=parent,
                domain=domain,
            )

            children_list.append(new_node)

        parent.update_children(children_list)

        if newlayer:
            self.node_list.append(children_list)
            self.depth += 1
        else:
            self.node_list[parent.get_depth() + 1] += children_list
