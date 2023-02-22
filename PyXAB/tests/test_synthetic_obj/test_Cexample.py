from PyXAB.synthetic_obj import Cexample
import pytest
import numpy as np


def test_Cexample_1():
    objective1 = Cexample.Cexample()
    assert objective1.fmax == 1


def test_Cexample_2():
    point = [0]
    objective1 = Cexample.Cexample()
    assert 0 == pytest.approx(objective1.f(point) - 1)


def test_Cexample_3():
    point = [1 / np.e]
    objective1 = Cexample.Cexample()
    assert 0 == pytest.approx(objective1.f(point))
