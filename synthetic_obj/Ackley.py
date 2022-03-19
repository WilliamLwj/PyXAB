import math
import numpy as np
import pdb

class Ackley:
    def __init__(self):

        self.fmax = 0

    def f(self, x):

        x1 = x[0]
        x2 = x[1]
        return 20 * np.exp(-0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2)))\
               + np.exp(0.5 * (np.cos(2*np.pi * x1) + np.cos(2*np.pi * x2))) - np.e - 20
