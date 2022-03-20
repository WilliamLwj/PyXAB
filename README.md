# PyXAB - Python *X*-Armed Bandit


<p align="left">
<a href="https://github.com/WilliamLwj/PyXAB/blob/master/LICENSE" target="blank">
<img src="https://img.shields.io/github/license/WilliamLwj/PyXAB?style=flat" alt="github-PyXAB license" />
</a>
<a href="https://github.com/WilliamLwj/PyXAB/fork" target="blank">
<img src="https://img.shields.io/github/forks/WilliamLwj/PyXAB?style=flat" alt="github-PyXAB forks"/>
</a>
<a href="https://github.com/WilliamLwj/PyXAB/stargazers" target="blank">
<img src="https://img.shields.io/github/stars/WilliamLwj/PyXAB?style=flat" alt="github-PyXAB stars"/>
</a>
</p>


Python implementation of *X*-armed bandit algorithms (also known as global optimization algorithms or bandit-based blackbox optimization algorithms)



    
## Implemented:

(1). ***X*-armed bandit algorithms**

| Algorithm | Research Paper (with publication venues) | Year |
| --- | --- | --- |
| [T-HOO](https://github.com/WilliamLwj/PyXAB/blob/main/algos/HOO.py) | [*X*-Armed Bandit](https://jmlr.org/papers/v12/bubeck11a.html) | 2011 |
| [HCT](https://github.com/WilliamLwj/PyXAB/blob/main/algos/HCT.py) | [Online Stochastic Optimization Under Correlated Bandit Feedback](https://proceedings.mlr.press/v32/azar14.html) | 2014 |
| [GPO](https://github.com/WilliamLwj/PyXAB/blob/main/algos/GPO.py) | [General Parallel Optimization Without A Metric](https://proceedings.mlr.press/v98/xuedong19a.html) | 2019 |
| [VHCT](https://github.com/WilliamLwj/PyXAB/blob/main/algos/VHCT.py) | [Optimum-statistical Collaboration Towards General and Efficient Black-box Optimization](https://arxiv.org/abs/2106.09215)  | 2021 |


(2). **Partition of the parameter space**

| Partition | Description |
| --- | --- |
| BinaryPartition | Equal-size binary partition of the parameter space, the split dimension is chosen uniform randomly|
| RandomBinaryPartition | The same as BinaryPartition but with a randomly chosen split point |
| DimensionBinaryPartition| Equal-size partition of the space with a binary split on each dimension, the number of children of one node is 2^d|

(3). **Synthetic Objectives** 


| Objectives| Mathematical Description | Image | 
| --- | --- |--- |
| Garland | <img src="https://render.githubusercontent.com/render/math?math=f(x) = x(1-x)(4-\sqrt{\mid\sin(60x)\mid})"> | <img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/synthetic/Garland.png" alt="Garland" width="100"/> |
| DoubleSine | <img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/synthetic/DoubleSine.png" alt="DoubleSine" width="100"/>  |
| DifficultFunc |  <img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/synthetic/DifficultFunc.png" alt="DifficultFunc" width="100"/>  |
| Ackley | <img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/synthetic/Ackley.png" alt="Ackley" width="100"/>  |
| Himmelblau | <img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/synthetic/Himmelblau.png" alt="Himmelblau" width="100"/>  |
| Rastrigin |  <img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/synthetic/Rastrigin.png" alt="Rastrigin" width="100"/>  |
