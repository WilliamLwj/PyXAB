import math
import numpy as np
import pdb

from partition.binary_partition import BinaryPartition

# domain = [[0, 1]]
# # part = BinaryPartition(domain)
# #
# # for i in range(5):
# #     part.deepen()
# #     nodelist = part.get_node_list()
# #     for node in nodelist[-1]:
# #         print(node.depth, node.index, node.domain, '\\')
# #
# #
# # node = part.get_node(4, 3)
# # print(node.depth, node.index, node.domain, '\\')
# # print(node.get_cpoint())




domain = [[0, 1], [10, 50], [-5, -10]]
part = BinaryPartition(domain)

for i in range(5):
    part.deepen()
    nodelist = part.get_node_list()
    for node in nodelist[-1]:
        print(node.depth, node.index, node.domain, '\\')


node = part.get_node(4, 3)
print(node.depth, node.index, node.domain, '\\')
print(node.get_cpoint())