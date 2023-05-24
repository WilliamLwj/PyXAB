General Instructions
===================================
To use PyXAB, the objective, the domain, and the hierarchical partition on the domain needs to be defined. Then the algorithms
evaluate the objective at one point in every round and receives a stochastic reward.


..................................

Domain
-------------

The domain needs to be written in list of lists for a continuous domain. For example,
if the parameter range is [0.01, 1], then the domain should be written as

.. code-block:: python3

    domain = [[0.01, 1]]

If the parameter has two dimensions, say [-1, 1] x [2, 10], then the domain should be written as

.. code-block:: python3

    domain = [[-1, 1], [2, 10]]

..................................

(Optional) Partition
-------------

The user can choose any designed partition, e.g., a binary partition would be

.. code-block:: python3

    from PyXAB.partition.BinaryPartition import BinaryPartition
    partition = BinaryPartition

By default, the standard binary partition will be used for all the algorithms

..................................

(Optional) Objective Function
-------------------------------
.. note::

    The objective function ``f`` should be bounded by -1 and 1 for the best performance of most algorithms, i.e., ``-1 <= f(x) <= 1``

.. note::

    It is unnecessary to define the objective function in the following way, but for consistency we recommend doing so. As long as
    the objective function can return a reward to the algorithm, then the optimization process could run.

The objective function has an attribute ``fmax``, which is the
maximum reward obtainable. Besides, the objective function
should have a function ``f(x)``, which will return the reward of the point ``x``.
See the following simple example for a better illustration.

.. code-block:: python3

    from PyXAB.synthetic_obj.Objective import Objective
    import numpy as np

    # The sine function f(x) = sin(x)
    class Sine(Objective):
        def __init__(self):
            self.fmax = 1

        def f(self, x):
            return np.sin(x)


..................................

Algorithm
-------------

Algorithms will always have one function named ``pull`` that outputs a point for evaluation, and the other function
named ``receive_reward`` to get the feedback. Therefore, in the online learning process, the following lines of code
should be used.


.. code-block:: python3

    from PyXAB.algos.HOO import T_HOO

    algo = T_HOO(domain=domain, partition=partition)
    target = Sine()

    # either for-loop or while-loop
    for t in range(1, T+1):
        point = algo.pull(t)
        reward = target.f(point) + np.random.uniform(-0.1, 0.1)
        algo.receive_reward(t, reward)


.. note::
    If the objective function is not defined by inheriting the :class:`PyXAB.synthetic_obj.Objective.Objective` class, simply change
    the second last line in the above snippet to the evaluation of the objective.
