
import math
import numpy as np
import pdb


class Himmelblau:

    def __init__(self):

        self.fmax = 0
    def f(self, x):

        x1 = x[0]
        x2 = x[1]
        return - (x1**2 + x2 - 11) ** 2 - (x1 + x2**2 - 7) ** 2
