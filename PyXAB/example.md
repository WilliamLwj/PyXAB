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


## Algorithm

Algorithms will always have one function named ```pull``` that outputs a point for evaluation, and the other function 
named ```receive_reward``` to get the feedback. Therefore, in the online learning process, the following lines of code
should be used.

```python3
# either for-loop or while-loop

for t in range(1, T+1):
    point = algo.pull(t)
    reward = target.f(point) + np.random.uniform(-0.1, 0.1)
    algo.receive_reward(t, reward)
```


