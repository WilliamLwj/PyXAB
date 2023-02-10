.. _api-cheatsheet:

API Cheatsheet
==========================


We list all the important functions from the general base classes as follows.

Algorithm
---------------------------

* :func:`PyXAB.algos.Algo.Algorithm.pull`: Generate a point for each time step to be evaluated
* :func:`PyXAB.algos.Algo.Algorithm.receive_reward`: After receiving the reward, update the parameters of the algorithm

Partition
---------------------------

* :func:`PyXAB.partition.Partition.Partition.make_children`: Make children for one node

Objective
---------------------------

* :func:`PyXAB.synthetic_obj.Objective.Objective.f`: Evaluate the point and return the reward (stochastic or deterministic)


...............................

Below are the general base classes

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
