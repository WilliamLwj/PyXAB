# PyXAB - Example

To use PyXAB, the objective, the domain, and the pre-defined partition on the domain needs to be defined. Then the algorithms
evaluate the objective at one point in every round and receives a stochastic reward.



## Objective Function
The objective function (class) needs to have an attribute ```fmax```, which is the
maximum reward obtainable, preferably ```fmax = 1```. Besides, the objective function 
should have a function ```f(x)```, which will return the reward of the point ```x```.
See the following simple example for a better illustration.
```python3
from PyXAB.synthetic_obj.Objective import Objective
import numpy as np

# The sine function f(x) = sin(x)
class Sine(Objective):
    def __init__(self):
        self.fmax = 1

    def f(self, x):    
        return np.sin(x)
```

## Domain
The domain needs to be written in list of lists for a continuous domain. For example,
if the parameter range is [0.01, 10], then the domain should be written as
```python3
domain = [[0.01, 10]]
```
If the parameter has two dimensions, say [-1, 1] x [2, 10], then the domain should be written as

```python3
domain = [[-1, 1], [2, 10]]
```

## Partition
The user can choose any designed partition, e.g., a binary partition would be

```Python3
from PyXAB.partition.BinaryPartition import BinaryPartition
partition = BinaryPartition
```


## A Complete Example - Running HCT on HimmelBlau

```python3
from PyXAB.synthetic_obj import *
from PyXAB.algos.HCT import HCT
from PyXAB.partition.BinaryPartition import BinaryPartition

import numpy as np
from PyXAB.utils.plot import plot_regret


# Define the number of rounds, target, domain, partition, and algorithm
T = 1000
target = HimmelBlau.Himmelblau()
domain = [[-5, 5], [-5, 5]]
partition = BinaryPartition
algo = HCT(domain=domain, partition=partition)


# regret and regret list
cumulative_regret = 0
cumulative_regret_list = []


# uniform noise
for t in range(1, T+1):

    point = algo.pull(t)
    reward = target.f(point) + np.random.uniform(-0.1, 0.1)
    algo.receive_reward(t, reward)
    inst_regret = target.fmax - target.f(point)
    cumulative_regret += inst_regret
    cumulative_regret_list.append(cumulative_regret)

# plot the regret
plot_regret(np.array(cumulative_regret_list))
```
