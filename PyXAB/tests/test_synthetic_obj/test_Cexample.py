from PyXAB.synthetic_obj import Cexample
import pytest
import numpy as np

def test_Cexample_value_error_1():
    point = [0, -1]  # An invalid point
    objective1 = Cexample.Cexample()
    with pytest.raises(ValueError):
        objective1.f(point)

def test_Cexample_value_error_2():
    point = [0, 1, -1]  # An invalid point
    objective1 = Cexample.Cexample()
    with pytest.raises(ValueError):
        objective1.f(point)



def test_Cexample_initialization():
    objective1 = Cexample.Cexample()
    assert objective1.fmax == 1


def test_Cexample_evaluation_1():
    point = [0]
    objective1 = Cexample.Cexample()
    assert 0 == pytest.approx(objective1.f(point) - 1)


def test_Cexample_evaluation_2():
    point = [1 / np.e]
    objective1 = Cexample.Cexample()
    assert 0 == pytest.approx(objective1.f(point))
