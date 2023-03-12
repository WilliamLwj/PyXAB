.. _api-cheatsheet:

API Cheatsheet
==========================


We list all the important (abstract) functions from the base classes as follows.

Algorithm
---------------------------

* :func:`PyXAB.algos.Algo.Algorithm.pull`: Generate a point for each time step to be evaluated
* :func:`PyXAB.algos.Algo.Algorithm.receive_reward`: After receiving the reward, update the parameters of the algorithm
* :func:`PyXAB.algos.Algo.Algorithm.get_last_point`: The function to retrieve the last output of the algorithm

Partition
---------------------------

* :func:`PyXAB.partition.Partition.Partition.make_children`: Make children for one node

Objective
---------------------------

* :func:`PyXAB.synthetic_obj.Objective.Objective.f`: Evaluate the point and return the reward (stochastic or deterministic)



.. note::

    The general base classes and the implemented functions are listed as follows

PyXAB.algos.Algo.Algorithm
---------------------------
Base class for all X-armed Bandit algorithms

.. autoclass:: PyXAB.algos.Algo.Algorithm
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__


PyXAB.synthetic_obj.Objective.Objective
---------------------------------------
Base class for any objective

.. autoclass:: PyXAB.synthetic_obj.Objective.Objective
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__



PyXAB.partition.Partition.Partition
------------------------------------
Base class for any partition


.. autoclass:: PyXAB.partition.Partition.Partition
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__



PyXAB.partition.Node.P_node
---------------------------
Base class for any node inside a partition

.. autoclass:: PyXAB.partition.Node.P_node
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__
