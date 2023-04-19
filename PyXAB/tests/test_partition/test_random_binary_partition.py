from PyXAB.partition.RandomBinaryPartition import RandomBinaryPartition
from numpy.testing import assert_allclose
import pytest


def test_random_binary_partition_value_error():
    with pytest.raises(ValueError):
        RandomBinaryPartition()


def test_random_binary_partition_1D_make_children_1():
    domain = [[0, 1]]
    part = RandomBinaryPartition(domain)

    parent = part.get_root()
    part.make_children(parent, newlayer=True)
    newlayer = part.get_node_list()[-1]
    assert (
        newlayer[0].get_domain()[0][1] == newlayer[1].get_domain()[0][0]
    )  # [[0, n]] and [[n, 1]]
    assert 1 >= newlayer[0].get_domain()[0][1] >= 0
    assert_allclose(newlayer[0].get_domain(), [[0, newlayer[0].get_domain()[0][1]]])
    assert_allclose(newlayer[1].get_domain(), [[newlayer[0].get_domain()[0][1], 1]])


def test_random_binary_partition_1D_make_children_2():
    domain = [[0, 1]]
    part = RandomBinaryPartition(domain)
    part.deepen()


def test_random_binary_partition_1D_deepen():
    domain = [[0, 1]]
    part = RandomBinaryPartition(domain)

    for i in range(5):
        part.deepen()
        nodelist = part.get_node_list()
        for node in nodelist[-1]:
            print(node.depth, node.index, node.domain, "\\")


def test_random_binary_partition_3D_deepen():
    domain = [[0, 1], [10, 50], [-5, -10]]
    part = RandomBinaryPartition(domain)

    for i in range(2):
        part.deepen()
        nodelist = part.get_node_list()
        for node in nodelist[-1]:
            print(node.depth, node.index, node.domain, "\\")
