
Algorithms
================================================


Our implemented *X*\ -Armed Bandit algorithms can be classified into different categories according to different
features in the algorithm design.


.. |check_| raw:: html

    &#x2714;

.. |cross_| raw:: html

    &#x2718;


.. list-table::
   :header-rows: 1

   * - Algorithm
     - Research
     - Stochastic
     - Cumulative
     - Anytime
   * - DiRect
     - paper
     - |cross_|
     - |cross_|
     - |cross_|
   * - `DOO <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/DOO.py>`_
     - `DOO paper <https://proceedings.neurips.cc/paper/2011/file/7e889fb76e0e07c11733550f2a6c7a5a-Paper.pdf>`_
     - |cross_|
     - |cross_|
     - |cross_|
   * - `SOO <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/SOO.py>`_
     - `SOO paper <https://proceedings.neurips.cc/paper/2011/file/7e889fb76e0e07c11733550f2a6c7a5a-Paper.pdf>`_
     - |cross_|
     - |cross_|
     - |cross_|
   * - `Zooming <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/Zooming.py>`_
     - `Zooming paper <https://arxiv.org/pdf/0809.4882.pdf>`_
     - |check_|
     - |check_|
     - |check_|
   * - `T-HOO <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/HOO.py>`_
     - `T-HOO paper <https://jmlr.org/papers/v12/bubeck11a.html>`_
     - |check_|
     - |check_|
     - |check_|
   * - `StoSOO <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/StoSOO.py>`_
     - `StoSOO paper <https://proceedings.mlr.press/v28/valko13.pdf>`_
     - |check_|
     - |cross_|
     - |cross_|
   * - `HCT <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/HCT.py>`_
     - `HCT paper <https://proceedings.mlr.press/v32/azar14.html>`_
     - |check_|
     - |check_|
     - |check_|
   * - `POO* <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/POO.py>`_
     - `POO paper <https://proceedings.neurips.cc/paper/2011/file/7e889fb76e0e07c11733550f2a6c7a5a-Paper.pdf>`_
     - |check_|
     - |cross_|
     - |check_|
   * - `GPO* <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/GPO.py>`_
     - `GPO paper <https://proceedings.mlr.press/v98/xuedong19a.html>`_
     - |check_|
     - |cross_|
     - |cross_|
   * - `PCT <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/PCT.py>`_
     - `GPO paper <https://proceedings.mlr.press/v98/xuedong19a.html>`_
     - |check_|
     - |cross_|
     - |cross_|
   * - `SequOOL <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/SequOOL.py>`_
     - `SequOOL paper <https://arxiv.org/pdf/1810.00997.pdf>`_
     - |cross_|
     - |cross_|
     - |cross_|
   * - `StroquOOL <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/StroquOOL.py>`_
     - `StroquOOL paper <https://arxiv.org/pdf/1810.00997.pdf>`_
     - |check_|
     - |cross_|
     - |cross_|
   * - `VROOM <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/VROOM.py>`_
     - `VROOM paper <https://arxiv.org/pdf/1910.04034.pdf>`_
     - |check_|
     - |cross_|
     - |cross_|
   * - `VHCT <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/VHCT.py>`_
     - `VHCT paper <https://openreview.net/forum?id=ClIcmwdlxn>`_
     - |check_|
     - |check_|
     - |check_|
   * - `VPCT <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/VPCT.py>`_
     - N.A.
     - |check_|
     - |cross_|
     - |cross_|


..................................


    * **(Stochastic)** For some algorithms such as T_HOO and HCT, they perform well in the stochastic *X*\ -Armed Bandit setting when there is noise in the problem. However for some of the algorithms, e.g., DOO, they only work in the noise-less (deterministic) setting.
    * **(Cumulative)** For some algorithms such as T_HOO and HCT, they are designed to optimize the cumulative regret, i.e., the performance over the whole learning process. However for algorithms such as StoSOO and StroquOOL, they will optimize the simple regret, i.e., the final-round/last output performance.
    * **(Anytime)** For some algorithms such as SequOOL and StroquOOL, they need the total number of rounds (budget) information to run the algorithm, but for algorithms such as T_HOO and HCT, they do not need such information.


.. note::
    Please refer to the following details for more information.



.. toctree::
    :maxdepth: 1

    algorithms/Zooming/Zooming
    algorithms/DOO/DOO
    algorithms/SOO/SOO
    algorithms/StoSOO/StoSOO
    algorithms/T-HOO/T-HOO
    algorithms/HCT/HCT
    algorithms/POO/POO
    algorithms/GPO/GPO
    algorithms/PCT/PCT
    algorithms/SequOOL/SequOOL
    algorithms/StroquOOL/StroquOOL
    algorithms/VROOM/VROOM
    algorithms/VHCT/VHCT
    algorithms/VPCT/VPCT




