Contributing
===================================

We appreciate all forms of help and contributions, including but not limited to

* Star and watch our project
* Open an issue for any bugs you find or features you want to add to our library
* Fork our project and submit a pull request with your valuable codes

.. note::
    We have some TODOs listed in the `Roadmap <https://github.com/users/WilliamLwj/projects/1/views/1>`_ that we need help with.


...........................

To Implement New Features
--------------------------
.. note::
    Please submit all pull requests to the ``dev`` branch instead of the ``main`` branch



**Before Implementation**

Please read the :ref:`api-cheatsheet` and :ref:`api-reference` for the API of our implemented classes and the abstract methods that need to be implemented when creating a new algorithm/partition/objective/node.

**During Implementation**

Please carefully follow our API and the :ref:`gallery_of_examples`. For example, every algorithm needs to inherit the class :class:`PyXAB.algos.Algo.Algorithm`, and has to implement the abstract methods :func:`PyXAB.algos.Algo.Algorithm.pull` and :func:`PyXAB.algos.Algo.Algorithm.receive_reward`.

**Documentations**

We do not ask for detailed documentations, but if it is possible, please add some comments and documentations for your implemented functions/classes, following the `numpy docstring <https://numpydoc.readthedocs.io/en/latest/format.html>`_ style.

**Testing and Debug**

After implementation, please test your algorithm by running it on some of our `synthetic objectives <https://pyxab.readthedocs.io/en/latest/api/functions.html>`_ for debugging and improvements and write a test_xxx.py file.

**Final Check**

Before submitting the pull request, please make sure you have the following files ready

.. code-block:: text

    xxx.py
    test_xxx.py


...............

Optional Steps
---------------
.. note::
    The following steps are optional but highly recommended

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


To obtain the testing results and the code coverage report, run the following lines

.. code-block:: bash

    coverage run --source=PyXAB -m pytest
    coverage report