from abc import ABC, abstractmethod


class Objective(ABC):
    """
    Abstract class for general blackbox objectives
    """

    @abstractmethod
    def f(self, x):
        """
        Evaluation of the chosen point

        Parameters
        ----------
        x: list
            one input point in the form of x = [x_1, x_2, ... x_d]

        Returns
        -------
        y: float
            Evaluated value of the function at the particular point
        """
        pass
