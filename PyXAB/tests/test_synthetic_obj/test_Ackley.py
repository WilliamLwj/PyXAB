from PyXAB.synthetic_obj import Ackley
import pytest


def test_Ackley_1():
    objective1 = Ackley.Ackley()
    objective2 = Ackley.Ackley_Normalized()
    assert objective1.fmax == 0
    assert objective2.fmax == 0


def test_Ackley_2():
    point = [0]  # An invalid point
    objective1 = Ackley.Ackley()
    objective2 = Ackley.Ackley_Normalized()
    with pytest.raises(ValueError):
        objective1.f(point)
    with pytest.raises(ValueError):
        objective2.f(point)


def test_Ackley_3():
    point = [0, 0]
    objective1 = Ackley.Ackley()
    objective2 = Ackley.Ackley_Normalized()
    assert 0 == pytest.approx(objective1.f(point))
    assert 0 == pytest.approx(objective2.f(point))
