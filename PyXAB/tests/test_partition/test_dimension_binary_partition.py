from PyXAB.partition.DimensionBinaryPartition import DimensionBinaryPartition
import pytest


def test_dimension_binary_partition_value_error():
    with pytest.raises(ValueError):
        DimensionBinaryPartition()


def test_dimension_binary_partition_1D_deepen():
    domain = [[0, 1]]
    part = DimensionBinaryPartition(domain)

    for i in range(5):
        part.deepen()
        nodelist = part.get_node_list()
        for node in nodelist[-1]:
            print(node.depth, node.index, node.domain, "\\")
    print(part.get_root().get_domain())


def test_dimension_binary_partition_3D_deepen():
    domain = [[0, 1], [10, 50], [-5, -10]]
    part = DimensionBinaryPartition(domain)

    for i in range(2):
        part.deepen()
        nodelist = part.get_node_list()
        for node in nodelist[-1]:
            print(node.depth, node.index, node.domain, "\\")
