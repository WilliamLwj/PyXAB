from PyXAB.partition.RandomBinaryPartition import RandomBinaryPartition
from PyXAB.partition.DimensionPartition import DimensionBinaryPartition

domain = [[0, 1]]
part = DimensionBinaryPartition(domain)

for i in range(5):
    part.deepen()
    nodelist = part.get_node_list()
    for node in nodelist[-1]:
        print(node.depth, node.index, node.domain, "\\")


domain = [[0, 1], [10, 50], [-5, -10]]
part = DimensionBinaryPartition(domain)

for i in range(2):
    part.deepen()
    nodelist = part.get_node_list()
    for node in nodelist[-1]:
        print(node.depth, node.index, node.domain, "\\")
