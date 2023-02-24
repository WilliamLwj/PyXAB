from PyXAB.synthetic_obj import DifficultFunc
import pytest

def test_DifficultFunc_1():
    point = [0, 1]  # An invalid point
    objective1 = DifficultFunc.DifficultFunc()
    with pytest.raises(ValueError):
        objective1.f(point)


def test_DifficultFunc_2():
    objective1 = DifficultFunc.DifficultFunc()
    assert objective1.fmax == 0.0


def test_DifficultFunc_3():
    point = [0.5]
    objective1 = DifficultFunc.DifficultFunc()
    assert 0 == pytest.approx(objective1.f(point))


def test_DifficultFunc_4():
    point = [1]
    objective1 = DifficultFunc.DifficultFunc()
    assert -0.7071 == pytest.approx(objective1.f(point), 1e-3)



def test_DifficultFunc_5():
    point = [0.9]
    objective1 = DifficultFunc.DifficultFunc()
    assert -0.6324 == pytest.approx(objective1.f(point), 1e-3)