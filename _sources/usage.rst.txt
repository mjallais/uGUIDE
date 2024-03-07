Usage
=====

.. _installation:

Installation
------------

To use μGUIDE, first install it using pip:

1. We recommend to use a  `conda <https://docs.conda.io/en/latest/miniconda.html>`_ virtual environment. If `conda` is installed on your machine, an environment for installing ``μGUIDE`` can be created as follows:

.. code-block:: console

   $ conda create -n uGUIDE_env python=3.8 && conda activate uGUIDE_env


2. Fork the repository and run the following command to clone it on your local machine:

.. code-block:: console

   (uGUIDE_env) $ git clone git@github.com:{YOUR_GITHUB_USERNAME}/uGUIDE.git


3. ``cd`` to ``μGUIDE`` directory and install it:

.. code-block:: console

   (uGUIDE_env) $ cd uGUIDE
   (uGUIDE_env) $ pip install .

This will also install the dependencies of ``μGUIDE``.


4. To check if the installation worked fine, you can run:

.. code-block:: console

   (uGUIDE_env) $ python -c 'import uGUIDE'


and it should not give any error message.