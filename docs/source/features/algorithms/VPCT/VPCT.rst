VPCT Algorithm
==============

Introduction
------------
`code <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/VPCT.py>`_


We introduce VPCT by combining VHCT with the GPO algorithm for a smoothness-agnostic algorithm

Algorithm Parameters
--------------------
    * `nu_max (float)` – parameter nu_max of the VPCT algorithm
    * `rho_max (float)` – parameter rho of the VPCT algorithm
    * `rounds (int)` - the number of rounds/budget
    * `domain (list(list))` – The domain of the objective to be optimized
    * `partition` – The partition choice of the algorithm. Default: BinaryPartition.


Usage Example
-------------

.. note::

    Make sure to use `get_last_point()` to get the final output


.. code-block:: python3

    from PyXAB.synthetic_obj.Garland import Garland
    from PyXAB.algos.VPCT import VPCT

    domain = [[0, 1]]               # Parameter is 1-D and between 0 and 1
    target = Garland()
    rounds = 1000
    algo = VHCT(rounds=rounds, domain=domain)

    for t in range(rounds):
        point = algo.pull(t)
        reward = target(point)
        algo.receive_reward(t, reward)

    algo.get_last_point()