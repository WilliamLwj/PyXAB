from PyXAB.partition.RandomKaryPartition import RandomKaryPartition
from numpy.testing import assert_allclose
import pytest


def test_random_kary_partition_value_error():
    with pytest.raises(ValueError):
        RandomKaryPartition()


def test_random_kary_partition_1D_K3_make_children():
    domain = [[0, 1]]
    part = RandomKaryPartition(domain, K=3)

    parent = part.get_root()
    part.make_children(parent, newlayer=True)
    newlayer = part.get_node_list()[-1]
    assert newlayer[0].get_domain()[0][1] == newlayer[1].get_domain()[0][0]      # [[0, n1]] and [[n1, n2]] and [[n2, 1]]
    assert newlayer[1].get_domain()[0][1] == newlayer[2].get_domain()[0][0]
    assert 1 >= newlayer[0].get_domain()[0][1] >= 0
    assert 1 >= newlayer[1].get_domain()[0][1] >= 0
    assert_allclose(newlayer[0].get_domain(), [[0, newlayer[0].get_domain()[0][1]]])
    assert_allclose(newlayer[1].get_domain(), [[newlayer[0].get_domain()[0][1], newlayer[1].get_domain()[0][1]]])
    assert_allclose(newlayer[2].get_domain(), [[newlayer[1].get_domain()[0][1], 1]])


def test_random_kary_partition_3D_deepen():
    domain = [[0, 1], [10, 50], [-5, -10]]
    part = RandomKaryPartition(domain)

    for i in range(2):
        part.deepen()
        nodelist = part.get_node_list()
        for node in nodelist[-1]:
            print(node.depth, node.index, node.domain, "\\")
