import math
import numpy as np
import pdb



### Implementation of partition

class p_node:

    def __init__(self, depth, index, parent, range):
        self.depth = depth
        self.index = index
        self.parent = parent
        self.children = None
        self.range = range

        point = []
        for x in self.range:

            point.append((x[0] + x[1]) / 2)# TODO: Different Domains Other Than Continuous Domains

        self.point = point

    def return_point(self):

        return self.point