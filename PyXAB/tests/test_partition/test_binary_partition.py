from PyXAB.partition.BinaryPartition import BinaryPartition
from numpy.testing import assert_allclose
import pytest
import copy


def test_binary_partition_value_error():
    with pytest.raises(ValueError):
        BinaryPartition()


def test_binary_partition_1D_make_children_1():
    domain = [[0, 1]]
    part = BinaryPartition(domain)

    parent = part.get_root()
    part.make_children(parent, newlayer=True)
    newlayer = part.get_node_list()[-1]
    for i in range(len(newlayer)):
        assert_allclose(newlayer[i].get_domain(), [[i / 2, (i + 1) / 2]])


def test_binary_partition_1D_make_children_2():
    domain = [[0, 1]]
    part = BinaryPartition(domain)
    part.deepen()

    nodes = part.get_node_list()[-1]
    part.make_children(nodes[0], newlayer=True)

    newlayer = part.get_node_list()[-1]
    assert_allclose(newlayer[0].get_domain(), [[0, 1 / 4]])
    assert_allclose(newlayer[1].get_domain(), [[1 / 4, 1 / 2]])

    part.make_children(nodes[1], newlayer=False)

    newlayer = part.get_node_list()[-1]
    assert_allclose(newlayer[2].get_domain(), [[1 / 2, 3 / 4]])
    assert_allclose(newlayer[3].get_domain(), [[3 / 4, 1]])


def test_binary_partition_1D_deepen():
    domain = [[0, 1]]
    part = BinaryPartition(domain)

    for i in range(5):
        part.deepen()
        nodelist = part.get_node_list()
        for j in range(len(nodelist[-1])):
            assert_allclose(
                nodelist[-1][j].get_domain(),
                [[j / (2 ** (i + 1)), (j + 1) / (2 ** (i + 1))]],
            )


def test_binary_partition_3D_deepen():
    domain = [[0, 1], [10, 50], [-5, -10]]
    part = BinaryPartition(domain)

    for i in range(5):
        part.deepen()
    nodelist = part.get_node_list()
    for depth in range(part.get_depth() - 1):
        for parent in nodelist[depth]:
            parent_domain = parent.get_domain()
            children = parent.get_children()
            child_domain_1 = children[0].get_domain()
            child_domain_2 = children[1].get_domain()

            for i in range(len(domain)):
                if (
                    parent_domain[i][1] != child_domain_1[i][1]
                    and parent_domain[i][0] != child_domain_2[i][0]
                ):
                    ground_truth_domain_1 = copy.deepcopy(parent_domain)
                    ground_truth_domain_1[i][0] = parent_domain[i][0]
                    ground_truth_domain_1[i][1] = (
                        parent_domain[i][0] + parent_domain[i][1]
                    ) / 2

                    assert_allclose(child_domain_1, ground_truth_domain_1)

                    ground_truth_domain_2 = copy.deepcopy(parent_domain)
                    ground_truth_domain_2[i][0] = (
                        parent_domain[i][0] + parent_domain[i][1]
                    ) / 2
                    ground_truth_domain_2[i][1] = parent_domain[i][1]
                    assert_allclose(child_domain_2, ground_truth_domain_2)
