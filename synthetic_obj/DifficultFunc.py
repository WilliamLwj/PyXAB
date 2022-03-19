import math
import numpy as np
import pdb

def threshold(x):

    if(x - np.floor(x) < 0.5):
        x = 0
    else:
        x = 1

    return x


class DifficultFunc:
    def __init__(self):
        self.fmax = 0.

    def f(self, x):
        x = x[0]
        y = np.abs(x-0.5)
        if y == 0:
            return 0
        else:
            return threshold(np.log(y)) * (np.sqrt(y) - y ** 2) - np.sqrt(y)

    def fmax(self):
        return self.fmax
