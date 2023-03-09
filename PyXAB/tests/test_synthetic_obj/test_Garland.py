from PyXAB.synthetic_obj import Garland
import pytest


def test_Garland_value_error_1():
    point = [0, 1]  # An invalid point
    objective1 = Garland.Garland()
    objective2 = Garland.Perturbed_Garland()
    with pytest.raises(ValueError):
        objective1.f(point)
    with pytest.raises(ValueError):
        objective2.f(point)


def test_Garland_value_error_2():
    point = [0, 1, 2]  # An invalid point
    objective1 = Garland.Garland()
    objective2 = Garland.Perturbed_Garland()
    with pytest.raises(ValueError):
        objective1.f(point)
    with pytest.raises(ValueError):
        objective2.f(point)


def test_Garland_initialization():
    objective1 = Garland.Garland()
    objective2 = Garland.Perturbed_Garland()
    assert objective1.fmax == 1.0
    assert objective2.fmax == pytest.approx(1.0 + objective2.perturb)


def test_Garland_evaluation():
    point = [0.5]
    objective1 = Garland.Garland()
    objective2 = Garland.Perturbed_Garland()
    assert 0.7515 == pytest.approx(objective1.f(point))
    assert 0.7515 == pytest.approx(objective2.f(point) - objective2.perturb)
