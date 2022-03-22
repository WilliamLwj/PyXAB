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


<p align='center'>
  <img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/partition.png" alt="Partition" width="45%"/>  
  <img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/HCT_visual.gif" alt="visualization" width="54%"/>  
</p>


## Quick Example
First define the blackbox objective, the parameter domain, the partition of the space, and the algorithm, e.g.

```python3
target = Garland()
domain = [[0, 1]]
partition = BinaryPartition
algo = T_HOO(rounds=T, domain=domain, partition=partition)
```

At every round  `t`, call `algo.pull(t)` to get a point. After receiving the (stochastic) reward for the point, call 
`algo.receive_reward(t, reward)` to give the algorithm the feedback

```python3
point = algo.pull(t)
reward = target.f(point) + np.random.uniform(-0.1, 0.1)
algo.receive_reward(t, reward)
```



## Citation
If you use our package in your research or projects, we kindly ask you to cite our work
```text
@article{li2021optimum,
  title={Optimum-statistical Collaboration Towards General and Efficient Black-box Optimization},
  author={Li, Wenjie and Wang, Chi-Hua, Qifan Song and Cheng, Guang},
  journal={arXiv preprint arXiv:2106.09215},
  year={2021}
}
```

    
## Implemented Features:

(1). ***X*-armed bandit algorithms**

* Algorithm starred are meta-algorithms (wrappers)

| Algorithm | Research Paper | Year |
| --- | --- | --- |
| [T-HOO](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/HOO.py) | [*X*-Armed Bandit](https://jmlr.org/papers/v12/bubeck11a.html) | 2011 |
| [HCT](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/HCT.py) | [Online Stochastic Optimization Under Correlated Bandit Feedback](https://proceedings.mlr.press/v32/azar14.html) | 2014 |
| [GPO*](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/GPO.py) | [General Parallel Optimization Without A Metric](https://proceedings.mlr.press/v98/xuedong19a.html) | 2019 |
| [PCT](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/GPO.py) | [General Parallel Optimization Without A Metric](https://proceedings.mlr.press/v98/xuedong19a.html) | 2019 |
| [VHCT](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/algos/VHCT.py) | [Optimum-statistical Collaboration Towards General and Efficient Black-box Optimization](https://arxiv.org/abs/2106.09215)  | 2021 |


(2). **Partition of the parameter space**

| Partition | Description |
| --- | --- |
| [BinaryPartition](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/partition/BinaryPartition.py) | Equal-size binary partition of the parameter space, the split dimension is chosen uniform randomly|
| [RandomBinaryPartition](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/partition/RandomBinaryPartition.py) | The same as BinaryPartition but with a randomly chosen split point |
| [DimensionBinaryPartition](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/partition/DimensionPartition.py)| Equal-size partition of the space with a binary split on each dimension, the number of children of one node is 2^d|

(3). **Synthetic Objectives**  

* Some of these objectives can be found [here](https://en.wikipedia.org/wiki/Test_functions_for_optimization)

| Objectives| Mathematical Description | Image | 
| --- | --- |--- |
| [Garland](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/synthetic_obj/Garland.py) | <img src="https://render.githubusercontent.com/render/math?math=f(x) = x(1-x)(4-\sqrt{\mid\sin(60x)\mid})"> | <img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/synthetic/Garland.png" alt="Garland" width="100"/> |
| [DoubleSine](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/synthetic_obj/DoubleSine.py)  |<img src="https://render.githubusercontent.com/render/math?math=f(x)=s(\frac{1}{2}\log_2 \mid 2x-1\mid)(\mid 2x-1\mid)^{-\log_2 \rho_2 } - (2x-1)^{-\log_2 \rho_1 }) - (\mid 2x-1\mid)^{-\log_2 \rho_1 }"> | <img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/synthetic/DoubleSine.png" alt="DoubleSine" width="100"/>  |
| [DifficultFunc](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/synthetic_obj/DifficultFunc.py) |  <img src="https://render.githubusercontent.com/render/math?math=f(x)=s(\log_2 \mid x-0.5\mid)(\sqrt{\mid x-0.5\mid} - (x-0.5)^2) - \sqrt{\mid x-0.5\mid} ">| <img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/synthetic/DifficultFunc.png" alt="DifficultFunc" width="100"/>  |
| [Ackley](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/synthetic_obj/Ackley.py) | <img src="https://render.githubusercontent.com/render/math?math=f(x,y) = 20 \exp \left[-0.2 \sqrt{0.5\left(x^{2}+y^{2}\right)}\right]-\exp [0.5(\cos 2 \pi x+\cos 2 \pi y)]-e-20">  | <img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/synthetic/Ackley.png" alt="Ackley" width="100"/>  |
| [Himmelblau](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/synthetic_obj/Himmelblau.py) |  <img src="https://render.githubusercontent.com/render/math?math=f(x, y)=-\left(x^{2}+y-11\right)^{2}-\left(x+y^{2}-7\right)^{2}">  | <img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/synthetic/Himmelblau.png" alt="Himmelblau" width="100"/>  |
| [Rastrigin](https://github.com/WilliamLwj/PyXAB/blob/main/PyXAB/synthetic_obj/Rastrigin.py) | <img src="https://render.githubusercontent.com/render/math?math=f(\mathbf{x})= - A n+\sum_{i=1}^{n}\left[x_{i}^{2} + A \cos \left(2 \pi x_{i}\right)\right]">  |  <img src="https://github.com/WilliamLwj/PyXAB/blob/main/figs/synthetic/Rastrigin.png" alt="Rastrigin" width="100"/>  |
