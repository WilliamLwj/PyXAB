import random
import math
import numpy as np
import pdb

def mysin(x):

    if(x - np.floor(x) < 0.5):
        x = 0
    else:
        x = 1

    return x

def mysin2(x):
    return (np.sin(x*2*np.pi)+1)/2.


def std_center(s):
    a, b = s
    return (a+b)/2.


def std_rpoint(s):
    a, b = s
    return a + (b-a)*random.random()


def std_split(s):
    a, b = s
    m = (a+b)/2.
    return [(a, m), (m, b)]


def std_noise(x, noise_lvl):
    return x + noise_lvl*random.random()


def noisyy(x):
    if x > random.random():
        return 1.
    else:
        return 0.



class DifficultFunc:
    def __init__(self):
        self.fmax = 0.

    def f(self, x):
        x = x[0]
        y = np.abs(x-0.5)
        if y == 0:
            return 0
        else:
            return mysin(np.log(y)) * (np.sqrt(y) - y ** 2) - np.sqrt(y)
    def fmax(self):
        return self.fmax


class DoubleSine:
    def __init__(self, rho1, rho2, tmax):
        self.ep1 = -math.log(rho1, 2)
        self.ep2 = -math.log(rho2, 2)
        self.tmax = tmax
        self.fmax = 0.

    def f(self, x):
        x = x[0]
        u = 2*np.fabs(x-self.tmax)
        if u == 0:
            return 0.
        else:
            envelope_width = math.pow(u, self.ep2)-math.pow(u, self.ep1)
            return mysin2(math.log(u, 2)/2.)*envelope_width - math.pow(u, self.ep2)

    def fmax(self):
        return self.fmax


class EasyFunc:
    def __init__(self):

        self.fmax = 0

    def f(self, x):
        x = x[0]
        return - x ** 2



class Cexample:
    def __init__(self):

        self.fmax = 1

    def f(self, x):
        x = x[0]
        return 1+1 / np.log(x)



class Garland:
    def __init__(self):

        self.fmax = 1

    def f(self, x):

        x = x[0]

        return x * (1-x) * (4 - np.sqrt(np.abs(np.sin(60 * x))))



class Himmelblau:

    def __init__(self):

        self.fmax = 0
    def f(self, x):

        x1 = x[0]
        x2 = x[1]
        return - (x1**2 + x2 - 11) ** 2 - (x1 + x2**2 - 7) ** 2


class Ackley:
    def __init__(self):

        self.fmax = 0
    def f(self, x):

        x1 = x[0]
        x2 = x[1]
        return 20 * np.exp(-0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2))) + np.exp(0.5 * (np.cos(2*np.pi * x1) + np.cos(2*np.pi * x2))) - np.e - 20



class Rastrigin:
    def __init__(self):

        self.fmax = 0
    def f(self, x):

        x = np.array(x)
        S = 0
        for i in range(x.size):
            S = S - 10 - (x[i]**2 - 10 * np.cos(2*np.pi * x[i]))
        return S


def plot_Garland():

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5), dpi=100)
    function = Garland()
    x = np.arange(0, 100000) * 1/100000
    val = []
    for i in x:
        val.append(function.f([i]))
    plt.plot(x, val)

    plt.show()
    #print(np.max(np.array(val)))


def plot_cexample():

    import matplotlib.pyplot as plt
    function = Cexample()
    x = np.arange(0, 100000) * 1/np.e /100000
    val = []
    for i in x:
        val.append(function.f([i]))
    plt.plot(x, val)
    plt.xlim((-0.01, 1/np.e))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.show()
    #print(np.max(np.array(val)))



def plot_DifficultFunc():

    import matplotlib.pyplot as plt
    function = DifficultFunc()
    x = np.arange(0, 100000) /100000
    val = []
    for i in x:
        val.append(function.f([i]))
    plt.plot(x, val)
    plt.xlim((0, 1))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.show()
    #print(np.max(np.array(val)))


def plot_DoubleSine():

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5), dpi=100)
    function = DoubleSine(0.3, 0.8, 0.5)
    x = np.arange(0, 100000) * 1/100000
    val = []
    for i in x:
        val.append(function.f([i]))
    plt.plot(x, val)
    plt.show()
    #print(np.max(np.array(val)))

def plot_Himmelblau():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    plt.figure(figsize=(6, 5), dpi=100)
    b = np.arange(-5, 5, 0.1)
    d = np.arange(-5, 5, 0.1)

    x1, x2 = np.meshgrid(b, d)
    y = - (x1**2 + x2 - 11) ** 2 - (x1 + x2**2 - 7) ** 2

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x1, x2, y)
    plt.xlabel('b')
    plt.ylabel('d')
    plt.show()

def plot_Ackley():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    b = np.arange(-1, 1, 0.1)
    d = np.arange(-1, 1, 0.1)

    x1, x2 = np.meshgrid(b, d)
    y = 20 * np.exp(-0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2))) + np.exp(0.5 * (np.cos(2*np.pi * x1) + np.cos(2*np.pi * x2))) - np.e - 20

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x1, x2, y)
    plt.xlabel('b')
    plt.ylabel('d')
    plt.show()



def plot_Rastrigin():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    b = np.arange(-5, 5, 0.1)
    d = np.arange(-5, 5, 0.1)

    x1, x2 = np.meshgrid(b, d)
    y = -20 - (x1**2 - 10 * np.cos(2*np.pi *x1))- (x2**2 - 10 * np.cos(2*np.pi *x2))
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x1, x2, y)
    plt.xlabel('b')
    plt.ylabel('d')
    plt.show()


#plot_cexample()
#plot_DifficultFunc()


# plot_Garland()
# plot_Himmelblau()
# plot_DoubleSine()
#plot_Rastrigin()
# plot_Ackley()



class constructed_function():

    def __init__(self):

        self.fmax = 0
    def f(self, x):

        x_repre = [1 / 8, 3 / 8, 5 / 8, 7 / 8, 1 / 16, 3 / 16, 5 / 16, 7 / 16, 9 / 16, 11 / 16, 13 / 16, 15 / 16]
        x_last = [1 / 32, 3 / 32, 5 / 32, 7 / 32, 9 / 32, 11 / 32, 13 / 32, 15 / 32, 17 / 32, 19 / 32, 21 / 32, 23 / 32,
                  25 / 32, 27 / 32, 29 / 32, 31 / 32]

        y_last = {1 / 8: 2 / 3, 3 / 8: 2 / 3, 5 / 8: 2 / 3, 7 / 8: 1 / 2,
                  1 / 16: 3 / 4, 3 / 16: 2 / 3, 5 / 16: 3 / 4, 7 / 16: 3 / 4, 9 / 16: 3 / 4, 11 / 16: 2 / 3,
                  13 / 16: 1 / 2, 15 / 16: 1 / 2,
                  1 / 32: 4 / 5, 3 / 32: 3 / 4, 5 / 32: 2 / 3, 7 / 32: 2 / 3, 9 / 32: 3 / 4, 11 / 32: 4 / 5,
                  13 / 32: 4 / 5, 15 / 32: 1, 17 / 32: 4 / 5, 19 / 32: 3 / 4, 21 / 32: 2 / 3, 23 / 32: 2 / 3,
                  25 / 32: 1 / 2, 27 / 32: 1 / 2, 29 / 32: 1 / 2, 31 / 32: 1 / 2}


        if x in x_repre or x in x_last:

            return y_last[x]

        else:
            if x < 2 / 32:
                return y_last[1 / 32]
            elif x < 4 / 32:
                return y_last[3 / 32]
            elif x < 6 / 32:
                return y_last[5 / 32]
            elif x < 8 / 32:
                return y_last[7 / 32]
            elif x < 10 / 32:
                return y_last[9 / 32]
            elif x < 12 / 32:
                return y_last[11 / 32]
            elif x < 14 / 32:
                return y_last[13 / 32]
            elif x < 16 / 32:
                return y_last[15 / 32]
            elif x < 18 / 32:
                return y_last[17 / 32]
            elif x < 20 / 32:
                return y_last[19 / 32]
            elif x < 22 / 32:
                return y_last[21 / 32]
            elif x < 24 / 32:
                return y_last[23 / 32]
            elif x < 26 / 32:
                return y_last[25 / 32]
            elif x < 28 / 32:
                return y_last[27 / 32]
            elif x < 30 / 32:
                return y_last[29 / 32]
            else:
                return y_last[31 / 32]


def plot_constructed():
    import matplotlib.pyplot as plt
    function = constructed_function()
    x = np.arange(0, 100000) / 100000
    val = []
    for i in x:
        val.append(function.f(i))
    plt.plot(x, val)
    plt.xlim((0, 1))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.show()


# plot_constructed()