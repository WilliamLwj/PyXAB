from PyXAB.synthetic_obj import Himmelblau
import pytest


def test_Himmelblau_1():
    objective1 = Himmelblau.Himmelblau()
    objective2 = Himmelblau.Himmelblau_Normalized()
    assert objective1.fmax == 0.0
    assert objective2.fmax == 0.0


def test_Himmelblau_2():
    point = [0, 0]
    objective1 = Himmelblau.Himmelblau()
    objective2 = Himmelblau.Himmelblau_Normalized()
    assert 0 == pytest.approx(objective1.f(point) + 170)
    assert 0 == pytest.approx(objective2.f(point) + 170 / 890)
