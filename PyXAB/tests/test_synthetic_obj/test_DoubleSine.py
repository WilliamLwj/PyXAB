from PyXAB.synthetic_obj import DoubleSine
import pytest



def test_DoubleSine_1():
    rho1 = 0
    rho2 = 0.8
    tmax = 0.5
    with pytest.raises(ValueError):
        DoubleSine.DoubleSine(rho1=rho1, rho2=rho2, tmax=tmax)
    with pytest.raises(ValueError):
        DoubleSine.Perturbed_DoubleSine(rho1=rho1, rho2=rho2, tmax=tmax)


def test_DoubleSine_2():
    rho1 = 1.2
    rho2 = 0.8
    tmax = 0.5
    with pytest.raises(ValueError):
        DoubleSine.DoubleSine(rho1=rho1, rho2=rho2, tmax=tmax)
    with pytest.raises(ValueError):
        DoubleSine.Perturbed_DoubleSine(rho1=rho1, rho2=rho2, tmax=tmax)

def test_DoubleSine_3():
    rho1 = 0.3
    rho2 = 0
    tmax = 0.5
    with pytest.raises(ValueError):
        DoubleSine.DoubleSine(rho1=rho1, rho2=rho2, tmax=tmax)
    with pytest.raises(ValueError):
        DoubleSine.Perturbed_DoubleSine(rho1=rho1, rho2=rho2, tmax=tmax)


def test_DoubleSine_4():
    rho1 = 0.3
    rho2 = 1.2
    tmax = 0.5
    with pytest.raises(ValueError):
        DoubleSine.DoubleSine(rho1=rho1, rho2=rho2, tmax=tmax)
    with pytest.raises(ValueError):
        DoubleSine.Perturbed_DoubleSine(rho1=rho1, rho2=rho2, tmax=tmax)

def test_DoubleSine_5():
    rho1 = 0.3
    rho2 = 0.8
    tmax = -0.1
    with pytest.raises(ValueError):
        DoubleSine.DoubleSine(rho1=rho1, rho2=rho2, tmax=tmax)
    with pytest.raises(ValueError):
        DoubleSine.Perturbed_DoubleSine(rho1=rho1, rho2=rho2, tmax=tmax)


def test_DoubleSine_6():
    rho1 = 0.3
    rho2 = 0.8
    tmax = 1.1
    with pytest.raises(ValueError):
        DoubleSine.DoubleSine(rho1=rho1, rho2=rho2, tmax=tmax)
    with pytest.raises(ValueError):
        DoubleSine.Perturbed_DoubleSine(rho1=rho1, rho2=rho2, tmax=tmax)


def test_DoubleSine_7():
    point = [0, 1]  # An invalid point
    objective1 = DoubleSine.DoubleSine()
    objective2 = DoubleSine.Perturbed_DoubleSine()
    with pytest.raises(ValueError):
        objective1.f(point)
    with pytest.raises(ValueError):
        objective2.f(point)


def test_DoubleSine_8():
    point = [0, 1, 0]  # An invalid point
    objective1 = DoubleSine.DoubleSine()
    objective2 = DoubleSine.Perturbed_DoubleSine()
    with pytest.raises(ValueError):
        objective1.f(point)
    with pytest.raises(ValueError):
        objective2.f(point)


def test_DoubleSine_9():
    objective1 = DoubleSine.DoubleSine()
    objective2 = DoubleSine.Perturbed_DoubleSine()
    assert objective1.fmax == 0.0
    assert objective2.fmax != 0.0


def test_DoubleSine_10():
    point = [0.5]
    objective1 = DoubleSine.DoubleSine()
    objective2 = DoubleSine.Perturbed_DoubleSine()
    assert 0 == pytest.approx(objective1.f(point))
    assert 0 == pytest.approx(objective2.f(point) - objective2.fmax)



def test_DoubleSine_11():
    point = [0]
    objective1 = DoubleSine.DoubleSine()
    objective2 = DoubleSine.Perturbed_DoubleSine()
    assert -1 == pytest.approx(objective1.f(point))
    assert -1 == pytest.approx(objective2.f(point) - objective2.fmax)