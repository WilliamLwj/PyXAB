Fixed Budget - Unknown Budget
================================================

For some algorithms such as SequOOL and StroquOOL, they need the total number of rounds (budget) information to run
the algorithm, but for algorithms such as T_HOO and HCT, they do not need such information.

.. note::
    Algorithm starred are meta-algorithms (wrappers)

Fixed Budget Algorithms
------------------------------------------------


.. list-table::
   :header-rows: 1

   * - Algorithm
     - Research Paper
     - Year
   * - DiRect
     - `Lipschitzian optimization without the Lipschitz constant <https://link.springer.com/article/10.1007/BF00941892>`_
     - 1993
   * - `DOO <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/DOO.py>`_
     - `Optimistic Optimization of a Deterministic Function without the Knowledge of its Smoothness <https://proceedings.neurips.cc/paper/2011/file/7e889fb76e0e07c11733550f2a6c7a5a-Paper.pdf>`_
     - 2011
   * - `SOO <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/SOO.py>`_
     - `Optimistic Optimization of a Deterministic Function without the Knowledge of its Smoothness <https://proceedings.neurips.cc/paper/2011/file/7e889fb76e0e07c11733550f2a6c7a5a-Paper.pdf>`_
     - 2011
   * - `StoSOO <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/StoSOO.py>`_
     - `Stochastic Simultaneous Optimistic Optimization <http://proceedings.mlr.press/v28/valko13.pdf>`_
     - 2013
   * - `POO* <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/POO.py>`_
     - `Black-box optimization of noisy functions with unknown smoothness <https://papers.nips.cc/paper/2015/hash/ab817c9349cf9c4f6877e1894a1faa00-Abstract.html>`_
     - 2015
   * - `GPO* <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/GPO.py>`_
     - `General Parallel Optimization Without A Metric <https://proceedings.mlr.press/v98/xuedong19a.html>`_
     - 2019
   * - `PCT <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/PCT.py>`_
     - `General Parallel Optimization Without A Metric <https://proceedings.mlr.press/v98/xuedong19a.html>`_
     - 2019
   * - `SequOOL <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/SequOOL.py>`_
     - `A Simple Parameter-free And Adaptive Approach to Optimization Under A Minimal Local Smoothness Assumption <https://arxiv.org/pdf/1810.00997.pdf>`_
     - 2019
   * - `StroquOOL <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/StroquOOL.py>`_
     - `A Simple Parameter-free And Adaptive Approach to Optimization Under A Minimal Local Smoothness Assumption <https://arxiv.org/pdf/1810.00997.pdf>`_
     - 2019
   * - `VPCT <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/VPCT.py>`_
     - N.A. (\ `GPO <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/GPO.py>`_ + `VHCT <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/VHCT.py>`_\ )
     - N.A.


...........................................

Anytime (Unknown Budget) Algorithms
------------------------------------------------


.. list-table::
   :header-rows: 1

   * - Algorithm
     - Research Paper
     - Year
   * - `Zooming <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/Zooming.py>`_
     - `Multi-Armed Bandits in Metric Spaces <https://arxiv.org/pdf/0809.4882.pdf>`_
     - 2008
   * - `T-HOO <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/HOO.py>`_
     - `\ *X*\ -Armed Bandit <https://jmlr.org/papers/v12/bubeck11a.html>`_
     - 2011
   * - `HCT <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/HCT.py>`_
     - `Online Stochastic Optimization Under Correlated Bandit Feedback <https://proceedings.mlr.press/v32/azar14.html>`_
     - 2014
   * - `VHCT <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/VHCT.py>`_
     - `Optimum-statistical Collaboration Towards General and Efficient Black-box Optimization <https://arxiv.org/abs/2106.09215>`_
     - 2021

