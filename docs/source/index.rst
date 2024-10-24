.. role:: raw-html-m2r(raw)
   :format: html


PyXAB - Python *X*\ -Armed Bandit
===================================

.. raw:: html

    <p align="left">
    <a style="border-width:0" href="https://doi.org/10.21105/joss.06507">
        <img src="https://joss.theoj.org/papers/10.21105/joss.06507/status.svg" alt="DOI badge" >
    </a>
    <a href="https://zenodo.org/doi/10.5281/zenodo.13743085">
        <img src="https://zenodo.org/badge/470722183.svg" alt="DOI" />
    </a>
    <a href='https://pypi.org/project/PyXAB/'>
           <img src='https://img.shields.io/pypi/v/PyXAB.svg?color=yellow' alt='PyPI version' />
    </a>
    <a href="https://codecov.io/gh/WilliamLwj/PyXAB" >
    <img src="https://codecov.io/gh/WilliamLwj/PyXAB/branch/main/graph/badge.svg?token=VACRX9AQBM"/>
    </a>
    <a href='https://pyxab.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/pyxab/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://github.com/WilliamLwj/PyXAB/actions/workflows/codeql.yml" target="blank">
    <img src="https://github.com/WilliamLwj/PyXAB/actions/workflows/codeql.yml/badge.svg" alt="Code style: black" />
    </a>
    <a href="https://github.com/WilliamLwj/PyXAB/actions/workflows/testing.yml" target="blank">
    <img src="https://github.com/WilliamLwj/PyXAB/actions/workflows/testing.yml/badge.svg" alt="testing" />
    </a>
    <a href="https://github.com/WilliamLwj/PyXAB/fork" target="blank">
    <img src="https://img.shields.io/github/forks/WilliamLwj/PyXAB?" alt="github-PyXAB forks"/>
    </a>
    <a href="https://github.com/WilliamLwj/PyXAB/stargazers" target="blank">
    <img src="https://img.shields.io/github/stars/WilliamLwj/PyXAB?" alt="github-PyXAB stars"/>
    </a>
    <a href="https://pepy.tech/project/pyxab" target="blank">
    <img src="https://static.pepy.tech/badge/pyxab" alt="downloads"/>
    </a>
    <a href="https://github.com/WilliamLwj/PyXAB/blob/main/LICENSE" target="blank">
    <img src="https://img.shields.io/github/license/WilliamLwj/PyXAB?color=purple" alt="github-PyXAB license" />
    </a>
    <a href="https://github.com/psf/black" target="blank">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" />
    </a>
    </p>

PyXAB is a Python open-source library for *X*-armed bandit algorithms, a prestigious set of optimizers for online black-box optimization and
hyperparameter optimization.

|pic1| |pic2|

.. |pic1| image:: https://raw.githubusercontent.com/WilliamLwj/PyXAB/main/figs/HCT_trajectory.gif
   :width: 48%
   :alt: Partition
.. |pic2| image:: https://raw.githubusercontent.com/WilliamLwj/PyXAB/main/figs/HCT_heatmap.gif
   :width: 48%
   :alt: visualization

PyXAB includes implementations of different algorithms for *X*\ -armed bandit, such as `Zooming <https://arxiv.org/pdf/0809.4882.pdf>`_\ ,
`StoSOO <https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/StoSOO.py>`_, and `HCT <https://proceedings.mlr.press/v32/azar14.html>`_\ , and the most
recent works such as `GPO <https://proceedings.mlr.press/v98/xuedong19a.html>`_ and `VHCT <https://arxiv.org/abs/2106.09215>`_.
PyXAB also provides the most commonly-used synthetic objectives to evaluate the performance of different algorithms and the implementations for different hierarchical partitions


**PyXAB is featured for:**

- **User-friendly APIs, clear documentation, and detailed examples**
- **Comprehensive library** of optimization algorithms, partitions and synthetic objectives
- **High standard code quality and high testing coverage**
- **Low dependency** for flexible combination with other packages such as PyTorch, Scikit-Learn

**Reminder**: The algorithms are maximization algorithms!



Quick Example
-------------
PyXAB follows a natural and straightforward API design completely aligned with the online blackbox
optimization paradigm. The following is a simple 6-line usage example.

First, we define the parameter domain and the algorithm to run.
At every round  ``t``, call ``algo.pull(t)`` to get a point and call
``algo.receive_reward(t, reward)`` to give the algorithm the objective evaluation (reward)

.. code-block:: python3

    domain = [[0, 1]]               # Parameter is 1-D and between 0 and 1
    algo = T_HOO(rounds=1000, domain=domain)
    for t in range(1000):
        point = algo.pull(t)
        reward = 1                  #TODO: User-defined objective returns the reward
        algo.receive_reward(t, reward)

Citations
---------



If you use our package in your research or projects, we kindly ask you to cite our work

.. code-block:: text

    @article{Li2023PyXAB,
        doi = {10.21105/joss.06507},
        url = {https://joss.theoj.org/papers/10.21105/joss.06507},
        author = {Li, Wenjie and Li, Haoze and Song, Qifan and Honorio, Jean},
        title = {PyXAB -- A Python Library for $\mathcal{X}$-Armed Bandit and Online Blackbox Optimization Algorithms},
        journal={Journal of Open Source Software},
        year = {2024},
        issn={2475-9066},
    }

We would appreciate it if you could cite our related works.

.. code-block:: text

    @article{li2023optimumstatistical,
        title={Optimum-statistical Collaboration Towards General and Efficient Black-box Optimization},
        author={Wenjie Li and Chi-Hua Wang and Guang Cheng and Qifan Song},
        journal={Transactions on Machine Learning Research},
        issn={2835-8856},
        year={2023},
        url={https://openreview.net/forum?id=ClIcmwdlxn},
        note={}
    }

.. code-block:: text

    @article{Li2024Federated,
        title={Federated $\chi$-armed Bandit}, 
        volume={38}, 
        url={https://ojs.aaai.org/index.php/AAAI/article/view/29267}, 
        DOI={10.1609/aaai.v38i12.29267}, 
        number={12},
        journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
        author={Li, Wenjie and Song, Qifan and Honorio, Jean and Lin, Guang}, 
        year={2024}, 
        month={Mar.}, 
        pages={13628-13636} 

        }

.. code-block:: text

    @InProceedings{Li2024Personalized,
        title = 	 {Personalized Federated $\chi$-armed Bandit},
        author =       {Li, Wenjie and Song, Qifan and Honorio, Jean},
        booktitle = 	 {Proceedings of The 27th International Conference on Artificial Intelligence and Statistics},
        pages = 	 {37--45},
        year = 	 {2024},
        editor = 	 {Dasgupta, Sanjoy and Mandt, Stephan and Li, Yingzhen},
        volume = 	 {238},
        series = 	 {Proceedings of Machine Learning Research},
        month = 	 {02--04 May},
        publisher =    {PMLR},
        pdf = 	 {https://proceedings.mlr.press/v238/li24a/li24a.pdf},
        url = 	 {https://proceedings.mlr.press/v238/li24a.html},
        }


.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Getting Started

   getting_started/installation
   getting_started/instructions
   getting_started/auto_examples/index



.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Features


   features/feat_algorithms
   features/feat_functions
   features/feat_partitions


.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: API

   api/cheatsheet
   api/index


.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Additional Info

   info/contributing
   info/auto_examples/index
   info/team_members