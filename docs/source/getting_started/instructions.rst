General Instructions
===================================
To use PyXAB, simply follow the instructions below. The domain and the algorithm must be defined beforehand. Hierarchical Partition
is optional and normally binary partition works well. The objective must be able to evaluate each point the algorithm pulls
and return the evaluated objective value.


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
--------------------
The hierarchical partition is a core part of many X-armed bandit algorithms. It discretizes the infinite parameter space into
finite number of arms in each layer hierarchically, so that finite-armed bandit algorithm designs can be utilized.

However, the design of the partition is completely optional and unnecessary in the experiments. PyXAB provides many designs
in the package for the users to choose from, e.g., a standard binary partition would be

.. code-block:: python3

    from PyXAB.partition.BinaryPartition import BinaryPartition
    partition = BinaryPartition

By default, the standard binary partition will be used for all the algorithms if unspecified.

..................................

(User Defined) Objective
-------------------------------
.. note::

    The objective function ``f`` should be bounded by -1 and 1 for the best performance of most algorithms, i.e., ``-1 <= f(x) <= 1``

.. note::

    It is unnecessary to define the objective function in the following way, but for consistency we recommend doing so. As long as
    the objective function can return a reward to each point pulled by the algorithm, then the optimization process could run.

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
            x = np.array(x)
            return np.sin(x)


..................................

Algorithm
-------------

.. note::

    The point returned by the algorithm will be a list. Make sure your objective can deal with this data type.
    For example, if it wants the objective value at the point x = 0.8, it will return [0.8]. If the algorithm wants the
    objective value at x = (0, 0.5), the algorithm will return [0, 0.5].

Algorithms will always have one function named ``pull`` that outputs a point for evaluation, and the other function
named ``receive_reward`` to get the feedback. Therefore, in the online learning process, the following lines of code
should be used.


.. code-block:: python3

    from PyXAB.algos.HOO import T_HOO
    T = 1000
    algo = T_HOO(rounds=T, domain=domain, partition=partition)
    target = Sine()

    # either for-loop or while-loop
    for t in range(1, T+1):
        point = algo.pull(t)
        reward = target.f(point) + np.random.uniform(-0.1, 0.1)
        algo.receive_reward(t, reward)


.. note::
    If the objective function is not defined by inheriting the :class:`PyXAB.synthetic_obj.Objective.Objective` class, simply change
    the second last line in the above snippet to the evaluation of the objective.
