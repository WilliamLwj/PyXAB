from PyXAB.synthetic_obj import DoubleSine
import pytest


def test_DoubleSine_1():
    objective1 = DoubleSine.DoubleSine()
    objective2 = DoubleSine.Perturbed_DoubleSine()
    assert objective1.fmax == 0.0
    assert objective2.fmax != 0.0


def test_DoubleSine_2():
    point = [0.5]
    objective1 = DoubleSine.DoubleSine()
    objective2 = DoubleSine.Perturbed_DoubleSine()
    assert 0 == pytest.approx(objective1.f(point))
    assert 0 == pytest.approx(objective2.f(point) - objective2.fmax)
