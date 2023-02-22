from PyXAB.synthetic_obj import Rastrigin
import pytest


def test_Rastrigin_1():
    objective1 = Rastrigin.Rastrigin()
    objective2 = Rastrigin.Rastrigin_Normalized()
    assert objective1.fmax == 0
    assert objective2.fmax == 0


def test_Rastrigin_2():
    point = [0, 0]
    objective1 = Rastrigin.Rastrigin()
    objective2 = Rastrigin.Rastrigin_Normalized()
    assert 0 == pytest.approx(objective1.f(point))
    assert 0 == pytest.approx(objective2.f(point))
