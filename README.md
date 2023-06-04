
# PyXAB - Python *X*-Armed Bandit 


<p align="left">
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


<p align='center'>
  <img src="https://raw.githubusercontent.com/WilliamLwj/PyXAB/main/figs/HCT_trajectory.gif" alt="trajectory" width="48%"/>  
  <img src="https://raw.githubusercontent.com/WilliamLwj/PyXAB/main/figs/HCT_heatmap.gif" alt="heatmap" width="48%"/>  
</p>

PyXAB contains the implementations of **10+** optimization algorithms, including the classic ones such as [Zooming](https://arxiv.org/pdf/0809.4882.pdf), 
 [StoSOO](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/StoSOO.py), and [HCT](https://proceedings.mlr.press/v32/azar14.html), and the most
recent works such as [GPO](https://proceedings.mlr.press/v98/xuedong19a.html), [StroquOOL](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/StroquOOL.py) and [VHCT](https://arxiv.org/abs/2106.09215).
PyXAB also provides the most commonly-used synthetic objectives to evaluate the performance of different algorithms and the implementations for different hierarchical partitions

**PyXAB is featured for:**

- **User-friendly APIs, clear documentation, and detailed examples**
- **Comprehensive library** of optimization algorithms, partitions and synthetic objectives
- **High standard code quality and testing coverage**


**Reminder**: The algorithms are maximization algorithms!

## Quick Links

- [Quick Example](#Quick-Example)
- [Documentations](#Documentations)
- [Installation](#Installation)
- [Features](#Features)
  * [*X*-armed bandit algorithms](#Stochastic-X-armed-bandit-algorithms)
  * [Hierarchical partition ](#Hierarchical-partition)
  * [Synthetic objectives](#Synthetic-objectives)
- [Contributing](#Contributing)
- [Citations](#Citations)

## Quick Example
PyXAB follows a natural and straightforward API design completely aligned with the online blackbox
optimization paradigm. The following is a simple 6-line usage example.

First, we define the parameter domain and the algorithm to run. 
At every round  `t`, call `algo.pull(t)` to get a point and call 
`algo.receive_reward(t, reward)` to give the algorithm the objective evaluation (reward)

```python3
domain = [[0, 1]]               # Parameter is 1-D and between 0 and 1
algo = T_HOO(rounds=1000, domain=domain) 
for t in range(1000):
    point = algo.pull(t)
    reward = 1                  #TODO: User-defined objective returns the reward
    algo.receive_reward(t, reward)
```

More detailed examples can be found [here](https://pyxab.readthedocs.io/en/latest/getting_started/auto_examples/index.html)

## Documentations

  * The most up-to-date [documentations](https://pyxab.readthedocs.io/) 

  * The [roadmap](https://github.com/users/WilliamLwj/projects/1) for our project    
  
  * Our [manuscript](https://arxiv.org/abs/2303.04030) for the library

## Installation

To install via pip, run the following lines of code
```bash
pip install PyXAB                 # normal install
pip install --upgrade PyXAB       # or update if needed
```


To install via git, run the following lines of code
```bash
git clone https://github.com/WilliamLwj/PyXAB.git
cd PyXAB
pip install .
```


## Features:

### *X*-armed bandit algorithms

* Algorithm starred are meta-algorithms (wrappers)

| Algorithm                                                                           | Research Paper                                                                                                                                                                           | Year |
|-------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
| [Zooming](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/Zooming.py)     | [Multi-Armed Bandits in Metric Spaces](https://arxiv.org/pdf/0809.4882.pdf)                                                                                                              | 2008 |
| [T-HOO](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/HOO.py)           | [*X*-Armed Bandit](https://jmlr.org/papers/v12/bubeck11a.html)                                                                                                                           | 2011 |
| [DOO](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/DOO.py)             | [Optimistic Optimization of a Deterministic Function without the Knowledge of its Smoothness](https://proceedings.neurips.cc/paper/2011/file/7e889fb76e0e07c11733550f2a6c7a5a-Paper.pdf) | 2011 |
| [SOO](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/SOO.py)             | [Optimistic Optimization of a Deterministic Function without the Knowledge of its Smoothness](https://proceedings.neurips.cc/paper/2011/file/7e889fb76e0e07c11733550f2a6c7a5a-Paper.pdf) | 2011 |
| [StoSOO](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/StoSOO.py)       | [Stochastic Simultaneous Optimistic Optimization](http://proceedings.mlr.press/v28/valko13.pdf)                                                                                          | 2013 |
| [HCT](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/HCT.py)             | [Online Stochastic Optimization Under Correlated Bandit Feedback](https://proceedings.mlr.press/v32/azar14.html)                                                                         | 2014 |
| [POO*](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/POO.py)            | [Black-box optimization of noisy functions with unknown smoothness](https://papers.nips.cc/paper/2015/hash/ab817c9349cf9c4f6877e1894a1faa00-Abstract.html)                               | 2015 |
| [GPO*](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/GPO.py)            | [General Parallel Optimization Without A Metric](https://proceedings.mlr.press/v98/xuedong19a.html)                                                                                      | 2019 |
| [PCT](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/PCT.py)             | [General Parallel Optimization Without A Metric](https://proceedings.mlr.press/v98/xuedong19a.html)                                                                                      | 2019 |
| [SequOOL](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/SequOOL.py)     | [A Simple Parameter-free And Adaptive Approach to Optimization Under A Minimal Local Smoothness Assumption](https://arxiv.org/pdf/1810.00997.pdf)                                        | 2019 |
| [StroquOOL](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/StroquOOL.py) | [A Simple Parameter-free And Adaptive Approach to Optimization Under A Minimal Local Smoothness Assumption](https://arxiv.org/pdf/1810.00997.pdf)                                        | 2019 |
| [VHCT](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/VHCT.py)           | [Optimum-statistical Collaboration Towards General and Efficient Black-box Optimization](https://arxiv.org/abs/2106.09215)                                                               | 2021 |
| [VPCT](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/VPCT.py)           | N.A. ([GPO](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/GPO.py) + [VHCT](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/VHCT.py))                               | N.A. |


### Hierarchical partition 

| Partition                                                                                                             | Description                                                                                                        |
|-----------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [BinaryPartition](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/partition/BinaryPartition.py)                   | Equal-size binary partition of the parameter space, the split dimension is chosen uniform randomly                 |
| [RandomBinaryPartition](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/partition/RandomBinaryPartition.py)       | Random-size binary partition of the parameter space, the split dimension is chosen uniform randomly                |
| [DimensionBinaryPartition](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/partition/DimensionBinaryPartition.py) | Equal-size partition of the space with a binary split on each dimension, the number of children of one node is 2^d |
| [KaryPartition](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/partition/KaryPartition.py)       | Equal-size K-ary partition of the parameter space, the split dimension is chosen uniform randomly                  |
| [RandomKaryPartition](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/partition/RandomKaryPartition.py) | Random-size K-ary partition of the parameter space, the split dimension is chosen uniform randomly                 |

### Synthetic objectives

* Some of these objectives can be found [on Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization)

| Objectives <img width=200/>| Image | 
| --- |--- |
| [Garland](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/synthetic_obj/Garland.py) | <img src="https://raw.githubusercontent.com/WilliamLwj/PyXAB/main/figs/synthetic/Garland.png" alt="Garland" width="100"/> |
| [DoubleSine](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/synthetic_obj/DoubleSine.py)  | <img src="https://raw.githubusercontent.com/WilliamLwj/PyXAB/main/figs/synthetic/DoubleSine.png" alt="DoubleSine" width="100"/>  |
| [DifficultFunc](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/synthetic_obj/DifficultFunc.py) | <img src="https://raw.githubusercontent.com/WilliamLwj/PyXAB/main/figs/synthetic/DifficultFunc.png" alt="DifficultFunc" width="100"/>  |
| [Ackley](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/synthetic_obj/Ackley.py) | <img src="https://raw.githubusercontent.com/WilliamLwj/PyXAB/main/figs/synthetic/Ackley.png" alt="Ackley" width="100"/>  |
| [Himmelblau](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/synthetic_obj/Himmelblau.py) | <img src="https://raw.githubusercontent.com/WilliamLwj/PyXAB/main/figs/synthetic/Himmelblau.png" alt="Himmelblau" width="100"/>  |
| [Rastrigin](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/synthetic_obj/Rastrigin.py) |  <img src="https://raw.githubusercontent.com/WilliamLwj/PyXAB/main/figs/synthetic/Rastrigin.png" alt="Rastrigin" width="100"/>  |


## Contributing

We appreciate all forms of help and contributions, including but not limited to

* Star and watch our project
* Open an issue for any bugs you find or features you want to add to our library
* Fork our project and submit a pull request with your valuable codes

Please read the [contributing instructions](https://pyxab.readthedocs.io/en/latest/info/contributing.html) before submitting
a pull request.


## Citations
If you use our package in your research or projects, we kindly ask you to cite our work

```text
@misc{Li2023PyXAB,
  doi = {10.48550/ARXIV.2303.04030},
  url = {https://arxiv.org/abs/2303.04030},
  author = {Li, Wenjie and Li, Haoze and Honorio, Jean and Song, Qifan},
  title = {PyXAB -- A Python Library for $\mathcal{X}$-Armed Bandit and Online Blackbox Optimization Algorithms},
  publisher = {arXiv},
  year = {2023},
}
```

We would also appreciate it if you could cite our related works.

```text
@article{li2023optimumstatistical,
    title={Optimum-statistical Collaboration Towards General and Efficient Black-box Optimization},
    author={Wenjie Li and Chi-Hua Wang and Guang Cheng and Qifan Song},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2023},
    url={https://openreview.net/forum?id=ClIcmwdlxn},
    note={}
}
```

```text
@misc{li2022Federated,
  doi = {10.48550/ARXIV.2205.15268},
  url = {https://arxiv.org/abs/2205.15268}, 
  author = {Li, Wenjie and Song, Qifan and Honorio, Jean and Lin, Guang},
  title = {Federated X-Armed Bandit},
  publisher = {arXiv},
  year = {2022},
}
```