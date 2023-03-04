
Algorithms
================================================


Our implemented *X*\ -Armed Bandit algorithms can be classified into different categories according to different features in the algorithm design.

.. |check_| raw:: html

    &#x2714;

.. |cross_| raw:: html

    &#x2718;


.. list-table::
   :header-rows: 1

   * - Algorithm
     - Stochastic
     - Cumulative Regret
     - Anytime
   * - DiRect
     - |cross_|
     - |cross_|
     - |cross_|
   * - `DOO <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/DOO.py>`_
     - |cross_|
     - |cross_|
     - |cross_|
   * - `SOO <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/SOO.py>`_
     - |cross_|
     - |cross_|
     - |cross_|
   * - Zooming
     - |check_|
     - |check_|
     - |check_|
   * - `T-HOO <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/HOO.py>`_
     - |check_|
     - |check_|
     - |check_|
   * - `StoSOO <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/StoSOO.py>`_
     - |check_|
     - |cross_|
     - |cross_|
   * - `HCT <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/HCT.py>`_
     - |check_|
     - |check_|
     - |check_|
   * - `POO* <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/POO.py>`_
     - |check_|
     - |cross_|
     - |check_|
   * - `GPO* <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/GPO.py>`_
     - |check_|
     - |cross_|
     - |cross_|
   * - `PCT <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/PCT.py>`_
     - |check_|
     - |cross_|
     - |cross_|
   * - `SequOOL <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/SequOOL.py>`_
     - |cross_|
     - |cross_|
     - |cross_|
   * - `StroquOOL <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/StroquOOL.py>`_
     - |check_|
     - |cross_|
     - |cross_|
   * - `VHCT <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/VHCT.py>`_
     - |check_|
     - |check_|
     - |check_|
   * - `VPCT <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/VPCT.py>`_
     - |check_|
     - |cross_|
     - |cross_|


..................................


.. note::
    Please refer to the following comparisons for more information.



.. toctree::
    :maxdepth: 3

    algorithms_category/stochasticity
    algorithms_category/regret
    algorithms_category/anytime




