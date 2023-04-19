from PyXAB.partition.DimensionBinaryPartition import DimensionBinaryPartition
from numpy.testing import assert_allclose
import pytest
import copy


def test_dimension_binary_partition_value_error():
    with pytest.raises(ValueError):
        DimensionBinaryPartition()


def test_dimension_binary_partition_1D_deepen():
    domain = [[0, 1]]
    part = DimensionBinaryPartition(domain)

    for i in range(5):
        part.deepen()
        nodelist = part.get_node_list()
        for j in range(len(nodelist[-1])):
            assert_allclose(
                nodelist[-1][j].get_domain(),
                [[j / (2 ** (i + 1)), (j + 1) / (2 ** (i + 1))]],
            )


def test_dimension_binary_partition_3D_deepen():
    domain = [[0, 1], [0, 1], [0, 1]]
    part = DimensionBinaryPartition(domain)
    for i in range(5):
        part.deepen()
    nodelist = part.get_node_list()

    for depth in range(part.get_depth() - 1):
        for parent in nodelist[depth]:
            parent_domain = parent.get_domain()
            children = parent.get_children()

            children_domain = []
            for child in children:
                children_domain.append(child.get_domain())
                print(child.get_depth(), child.get_index(), child.get_domain(), "\\")
