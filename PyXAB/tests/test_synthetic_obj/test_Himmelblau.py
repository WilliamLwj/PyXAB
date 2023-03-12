from PyXAB.synthetic_obj import Himmelblau
import pytest


def test_Himmelblau_value_error_1():
    point = [0]
    objective1 = Himmelblau.Himmelblau()
    objective2 = Himmelblau.Himmelblau_Normalized()
    with pytest.raises(ValueError):
        objective1.f(point)
    with pytest.raises(ValueError):
        objective2.f(point)


def test_Himmelblau_value_error_2():
    point = [0, 1, 2]
    objective1 = Himmelblau.Himmelblau()
    objective2 = Himmelblau.Himmelblau_Normalized()
    with pytest.raises(ValueError):
        objective1.f(point)
    with pytest.raises(ValueError):
        objective2.f(point)


def test_Himmelblau_initialization_fmax():
    objective1 = Himmelblau.Himmelblau()
    objective2 = Himmelblau.Himmelblau_Normalized()
    assert objective1.fmax == 0.0
    assert objective2.fmax == 0.0


def test_Himmelblau_evaluation():
    point = [0, 0]
    objective1 = Himmelblau.Himmelblau()
    objective2 = Himmelblau.Himmelblau_Normalized()
    assert 0 == pytest.approx(objective1.f(point) + 170)
    assert 0 == pytest.approx(objective2.f(point) + 170 / 890)
