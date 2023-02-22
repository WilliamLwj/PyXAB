from PyXAB.synthetic_obj import DifficultFunc
import pytest


def test_DifficultFunc_1():
    objective1 = DifficultFunc.DifficultFunc()
    assert objective1.fmax == 0.0


def test_DifficultFunc_2():
    point = [0.5]
    objective1 = DifficultFunc.DifficultFunc()
    assert 0 == pytest.approx(objective1.f(point))


def test_DifficultFunc_3():
    point = [1]
    objective1 = DifficultFunc.DifficultFunc()
    assert -0.7071 == pytest.approx(objective1.f(point), 1e-3)
