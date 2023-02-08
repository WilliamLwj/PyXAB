Hierarchical Partition
================================================

We provide several choices of the hierarchical partition that separates the parameter space into multiple pieces.


|pic1| |pic2|

.. |pic1| image:: https://raw.githubusercontent.com/WilliamLwj/PyXAB/main/figs/partition.png
    :width: 48%
    :alt: HCT Algorithm
.. |pic2| image:: https://raw.githubusercontent.com/WilliamLwj/PyXAB/main/figs/HCT_visual.gif
    :width: 48%
    :alt: HCT Algorithm



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
