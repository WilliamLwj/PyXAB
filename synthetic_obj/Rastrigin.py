import math
import numpy as np
import pdb


class Rastrigin:
    def __init__(self):

        self.fmax = 0

    def f(self, x):

        x = np.array(x)
        S = 0
        for i in range(x.size):
            S = S - 10 - (x[i]**2 - 10 * np.cos(2*np.pi * x[i]))
        return S
