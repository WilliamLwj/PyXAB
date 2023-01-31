Hierarchical Partition
================================================

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
