Contributing
===================================

We appreciate all forms of help and contributions, including but not limited to

* Star and watch our project
* Open an issue for any bugs you find or features you want to add to our library
* Fork our project and submit a pull request with your valuable codes

...........................

For Enhancements
--------------------------

Please see the :ref:`api-cheatsheet` and :ref:`api-reference` for the API of our implemented classes and the abstract methods
that need to be implemented when creating a new algorithm/partition/objective/node.

For example, every algorithm needs to inherit the class :class:`PyXAB.algos.Algo.Algorithm`, and has to implement
the abstract methods :func:`PyXAB.algos.Algo.Algorithm.pull` and :func:`PyXAB.algos.Algo.Algorithm.receive_reward`.

After implementation, please test your algorithm by running it on some of our
`synthetic objectives <https://pyxab.readthedocs.io/en/latest/api/functions.html>`_ for debugging and improvements.



...............

Optional
---------------

Black CodeStyle
^^^^^^^^^^^^^^^

In PyXAB, we follow the black codestyle. See more details `on webpage of black <https://github.com/psf/black>`_ and
`our issue <https://github.com/WilliamLwj/PyXAB/issues/14>`_. To convert your code, simply follow the instructions below.

First, run the following lines of code to install black

.. code-block:: bash

    python -m pip install --upgrade pip
    python -m pip install black

After implementing your own classes with documentations, run the following lines to change your code style

.. code-block:: bash

    black PyXAB
    python -m black PyXAB #if the above line does not work

..........................

Local Testing and Coverage
^^^^^^^^^^^^^^^^^^^^^^^^^^

First, run the following lines of code to install pytest and coverage

.. code-block:: bash

    python -m pip install --upgrade pip
    python -m pip install pytest==7.1.2
    python -m pip install coverage


To obtain the code coverage report, run the following lines

.. code-block:: bash

    coverage run --source=PyXAB -m pytest