import math
import numpy as np
import pdb


class Algorithm:

    def __init__(self, partition):

        self.partition = partition

    def pull(self, time):

        # Every algorithm must re-write this function

        raise NotImplementedError

    def receive_reward(self, time, reward):

        # Every algorithm must re-write this function

        raise NotImplementedError

