# -*- coding: utf-8 -*-
"""
New Partition
=====================
An example to implement a new partition of the space for any PyXAB algorithm. First import all the useful packages
"""

from PyXAB.synthetic_obj.Garland import Garland
from PyXAB.algos.HOO import T_HOO
from PyXAB.partition.Partition import Partition
from PyXAB.partition.Node import P_node
import numpy as np




# %%
# Now let us suppose that we want to implement a new binary partition that always split the domain into
# two nodes that are 1/3 and 2/3 of its original size, i.e., if the original projection on the chosen dimension is
# ``[a, b]``, we split the domain into ``[a, 0.67a + 0.33b]`` and ``[0.67a + 0.33b, b]``.

class NewBinaryPartition(Partition):

    def __init__(self, domain, node=P_node):

        super(NewBinaryPartition, self).__init__(domain=domain, node=node)

    # We rewrite the make_chilren function for the new partition
    def make_children(self, parent, newlayer=False):

        parent_domain = parent.get_domain()
        dim = np.random.randint(0, len(parent_domain))
        selected_dim = parent_domain[dim]

        domain1 = parent_domain.copy()
        domain2 = parent_domain.copy()

        # New choice of the split point
        split_point = 2/3 * selected_dim[0] + 1/3 * selected_dim[1]             # split point
        domain1[dim] = [selected_dim[0], split_point]
        domain2[dim] = [split_point, selected_dim[1]]

        # Initialization of the two new nodes
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

        # Update the children of the parent
        parent.update_children([node1, node2])

        new_deepest = []
        new_deepest.append(node1)
        new_deepest.append(node2)

        # If creating a new layer, use the new nodes as the first nodes in the new layer
        if newlayer:
            self.node_list.append(new_deepest)
            self.depth += 1
        # Else, just append the new nodes to the existing layer
        else:
            self.node_list[parent.get_depth() + 1] += new_deepest




# %%
# Define the number of rounds, the target, the domain, the partition, and the algorithm for the learning process
T = 100
target = Garland()
domain = [[0, 1]]
partition = NewBinaryPartition                      # the new partition chosen is NewBinaryPartition
algo = T_HOO(domain=domain, partition=partition)


# %%
# As shown below, the partition should be working

for t in range(1, T+1):

    point = algo.pull(t)
    reward = target.f(point) + np.random.uniform(-0.1, 0.1)     # uniform noise
    algo.receive_reward(t, reward)
