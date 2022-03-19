import math
import numpy as np
import pdb
from abc import ABC, abstractmethod


class Algorithm(ABC):

    def __init__(self, partition):

        self.partition = partition

    @abstractmethod
    def pull(self, time):

        # Every algorithm must re-write this function

        pass

    @abstractmethod
    def receive_reward(self, time, reward):

        # Every algorithm must re-write this function

        pass

