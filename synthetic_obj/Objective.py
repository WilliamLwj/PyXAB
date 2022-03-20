from abc import ABC, abstractmethod



class Objective(ABC):

    @abstractmethod
    def f(self, x):

        pass
