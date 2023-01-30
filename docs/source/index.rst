.. role:: raw-html-m2r(raw)
   :format: html


PyXAB - Python *X*\ -Armed Bandit
===================================


.. raw:: html

   <p align="left">
   <a href='https://pyxab.readthedocs.io/en/latest/?badge=latest'>
       <img src='https://readthedocs.org/projects/pyxab/badge/?version=latest' alt='Documentation Status' />
   </a>
   <a href="https://github.com/WilliamLwj/PyXAB/blob/main/LICENSE" target="blank">
   <img src="https://img.shields.io/github/license/WilliamLwj/PyXAB?style=flat-square" alt="github-PyXAB license" />
   </a>
   <a href="https://github.com/WilliamLwj/PyXAB/fork" target="blank">
   <img src="https://img.shields.io/github/forks/WilliamLwj/PyXAB?style=flat-square" alt="github-PyXAB forks"/>
   </a>
   <a href="https://github.com/WilliamLwj/PyXAB/stargazers" target="blank">
   <img src="https://img.shields.io/github/stars/WilliamLwj/PyXAB?style=flat-square" alt="github-PyXAB stars"/>
   </a>
   </p>


PyXAB is a Python open-source library for *X*\ -armed bandit, also known as continuous-arm bandit (CAB), Lipschitz bandit,
global optimization (GO) and bandit-based blackbox optimization problems.

PyXAB includes implementations of different algorithms for *X*\ -armed bandit, such as `HOO(Bubeck et al., 2011) <https://jmlr.org/papers/v12/bubeck11a.html>`_\ ,
`HCT(Azar et al., 2014) <https://proceedings.mlr.press/v32/azar14.html>`_\ , and the most
recent works `GPO(Shang et al., 2019) <https://proceedings.mlr.press/v98/xuedong19a.html>`_ and `VHCT(Li et al, 2021) <https://arxiv.org/abs/2106.09215>`_.
PyXAB also provides the most commonly-used synthetic objective that evaluate the performance of different algorithms such as Garland and DoubleSine.


.. raw:: html

   <p align='center'>
     <img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/partition.png" alt="Partition" width="45%"/>
     <img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/HCT_visual.gif" alt="visualization" width="54%"/>
   </p>


Quick Links
-----------


* `Quick Example <#Quick-Example>`_
* `Documentations <#Documentations>`_
* `Features <#Features>`_

  * `\ *X*\ -armed bandit algorithms <#X-armed-bandit-algorithms>`_
  * `Hierarchical partition  <#Hierarchical-partition>`_
  * `Synthetic objectives <#Synthetic-objectives>`_

* `Citations <#Citations>`_

Quick Example
-------------

First define the blackbox objective, the parameter domain, the partition of the space, and the algorithm, e.g.

.. code-block:: python3

   target = Garland()
   domain = [[0, 1]]
   partition = BinaryPartition
   algo = T_HOO(rounds=1000, domain=domain, partition=partition)

At every round  ``t``\ , call ``algo.pull(t)`` to get a point. After receiving the (stochastic) reward for the point, call
``algo.receive_reward(t, reward)`` to give the algorithm the feedback

.. code-block:: python3

   point = algo.pull(t)
   reward = target.f(point) + np.random.uniform(-0.1, 0.1) # Uniform noise example
   algo.receive_reward(t, reward)

Features:
---------

*X*\ -armed bandit algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* Algorithm starred are meta-algorithms (wrappers)

.. list-table::
   :header-rows: 1

   * - Algorithm
     - Research Paper
     - Year
   * - `T-HOO <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/HOO.py>`_
     - `\ *X*\ -Armed Bandit <https://jmlr.org/papers/v12/bubeck11a.html>`_
     - 2011
   * - `HCT <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/HCT.py>`_
     - `Online Stochastic Optimization Under Correlated Bandit Feedback <https://proceedings.mlr.press/v32/azar14.html>`_
     - 2014
   * - `POO* <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/POO.py>`_
     - `Black-box optimization of noisy functions with unknown smoothness <https://papers.nips.cc/paper/2015/hash/ab817c9349cf9c4f6877e1894a1faa00-Abstract.html>`_
     - 2015
   * - `GPO* <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/GPO.py>`_
     - `General Parallel Optimization Without A Metric <https://proceedings.mlr.press/v98/xuedong19a.html>`_
     - 2019
   * - `PCT <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/PCT.py>`_
     - `General Parallel Optimization Without A Metric <https://proceedings.mlr.press/v98/xuedong19a.html>`_
     - 2019
   * - `VHCT <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/VHCT.py>`_
     - `Optimum-statistical Collaboration Towards General and Efficient Black-box Optimization <https://arxiv.org/abs/2106.09215>`_
     - 2021
   * - `VPCT <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/VPCT.py>`_
     - N.A. (\ `GPO <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/GPO.py>`_ + `VHCT <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/VHCT.py>`_\ )
     - N.A.


Hierarchical partition
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Partition
     - Description
   * - `BinaryPartition <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/partition/BinaryPartition.py>`_
     - Equal-size binary partition of the parameter space, the split dimension is chosen uniform randomly
   * - `RandomBinaryPartition <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/partition/RandomBinaryPartition.py>`_
     - The same as BinaryPartition but with a randomly chosen split point
   * - `DimensionBinaryPartition <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/partition/DimensionPartition.py>`_
     - Equal-size partition of the space with a binary split on each dimension, the number of children of one node is 2^d


Synthetic objectives
^^^^^^^^^^^^^^^^^^^^


* Some of these objectives can be found `here <https://en.wikipedia.org/wiki/Test_functions_for_optimization>`_

.. list-table::
   :header-rows: 1

   * - Objectives
     - Mathematical Description
     - Image
   * - `Garland <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/synthetic_obj/Garland.py>`_
     - :raw-html-m2r:`<img src="https://render.githubusercontent.com/render/math?math=f(x) = x(1-x)(4-\sqrt{\mid\sin(60x)\mid})">`
     - :raw-html-m2r:`<img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/synthetic/Garland.png" alt="Garland" width="100"/>`
   * - `DoubleSine <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/synthetic_obj/DoubleSine.py>`_
     - :raw-html-m2r:`<img src="https://render.githubusercontent.com/render/math?math=f(x)=s(\frac{1}{2}\log_2 \mid 2x-1\mid)(\mid 2x-1\mid^{-\log_2 \rho_2 } - (2x-1)^{-\log_2 \rho_1 }) - (\mid 2x-1\mid)^{-\log_2 \rho_1 }">`
     - :raw-html-m2r:`<img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/synthetic/DoubleSine.png" alt="DoubleSine" width="100"/>`
   * - `DifficultFunc <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/synthetic_obj/DifficultFunc.py>`_
     - :raw-html-m2r:`<img src="https://render.githubusercontent.com/render/math?math=f(x)=s(\log_2 \mid x-0.5\mid)(\sqrt{\mid x-0.5\mid} - (x-0.5)^2) - \sqrt{\mid x-0.5\mid} ">`
     - :raw-html-m2r:`<img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/synthetic/DifficultFunc.png" alt="DifficultFunc" width="100"/>`
   * - `Ackley <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/synthetic_obj/Ackley.py>`_
     - :raw-html-m2r:`<img src="https://render.githubusercontent.com/render/math?math=f(x,y) = 20 \exp \left[-0.2 \sqrt{0.5\left(x^{2}-(-y^{2})\right)}\right]-\exp [0.5(\cos 2 \pi x-(-\cos 2 \pi y))]-e-20">`
     - :raw-html-m2r:`<img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/synthetic/Ackley.png" alt="Ackley" width="100"/>`
   * - `Himmelblau <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/synthetic_obj/Himmelblau.py>`_
     - :raw-html-m2r:`<img src="https://render.githubusercontent.com/render/math?math=f(x, y)=-\left(x^{2}-(-y)-11\right)^{2}-\left(x-(-y^{2})-7\right)^{2}">`
     - :raw-html-m2r:`<img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/synthetic/Himmelblau.png" alt="Himmelblau" width="100"/>`
   * - `Rastrigin <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/synthetic_obj/Rastrigin.py>`_
     - :raw-html-m2r:`<img src="https://render.githubusercontent.com/render/math?math=f(\mathbf{x})= - A n - \sum_{i=1}^{n}\left[-x_{i}^{2} - A \cos \left(2 \pi x_{i}\right)\right]">`
     - :raw-html-m2r:`<img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/synthetic/Rastrigin.png" alt="Rastrigin" width="100"/>`


Citations
---------

If you use our package in your research or projects, we kindly ask you to cite our work

.. code-block:: text

   @article{li2021optimum,
     title={Optimum-statistical Collaboration Towards General and Efficient Black-box Optimization},
     author={Li, Wenjie and Wang, Chi-Hua, Qifan Song and Cheng, Guang},
     journal={arXiv preprint arXiv:2106.09215},
     year={2021}
   }

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents:


   api/index