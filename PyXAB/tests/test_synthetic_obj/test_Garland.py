from PyXAB.synthetic_obj import Garland
import pytest


def test_Garland_1():
    objective1 = Garland.Garland()
    objective2 = Garland.Perturbed_Garland()
    assert objective1.fmax == 1.0
    assert objective2.fmax == pytest.approx(1.0 + objective2.perturb)


def test_Garland_2():
    point = [0.5]
    objective1 = Garland.Garland()
    objective2 = Garland.Perturbed_Garland()
    assert 0.7515 == pytest.approx(objective1.f(point))
    assert 0.7515 == pytest.approx(objective2.f(point) - objective2.perturb)



