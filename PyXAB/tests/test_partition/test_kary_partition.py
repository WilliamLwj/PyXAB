from PyXAB.partition.KaryPartition import KaryPartition
from numpy.testing import assert_allclose
import pytest


def test_kary_partition_value_error():
    with pytest.raises(ValueError):
        KaryPartition()


def test_kary_partition_1D_K3_make_children():
    domain = [[0, 1]]
    part = KaryPartition(domain)

    parent = part.get_root()
    part.make_children(parent, newlayer=True)
    newlayer = part.get_node_list()[-1]
    for i in range(len(newlayer)):
        assert_allclose(newlayer[i].get_domain(), [[i / 3, (i + 1) / 3]])
    print(part.get_root().get_domain())


def test_kary_partition_1D_K5_make_children():
    domain = [[0, 1]]
    K = 5
    part = KaryPartition(domain, K=K)

    parent = part.get_root()
    part.make_children(parent, newlayer=True)
    newlayer = part.get_node_list()[-1]
    for i in range(len(newlayer)):
        assert_allclose(newlayer[i].get_domain(), [[i / K, (i + 1) / K]])


def test_kary_partition_3D_K3_deepen():
    domain = [[0, 1], [10, 50], [-5, -10]]
    part = KaryPartition(domain)

    for i in range(2):
        part.deepen()
        nodelist = part.get_node_list()
        for node in nodelist[-1]:
            print(node.depth, node.index, node.domain, "\\")
