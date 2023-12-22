# μGUIDE

`μGUIDE` is a python package for estimating tissue microstructure from diffusion MRI acquisitions using simulation-based inference.
`μGUIDE` relies on a Bayesian approach for estimating posterior distributions.

## Installation
We recommend to use a [`conda`](https://docs.conda.io/en/latest/miniconda.html) virtual environment. An environment for installing `μGUIDE` can be created as follows:

```commandline
# Create an environment for μGUIDE and activate it
$ conda create -n uGUIDE_env python=3.8 && conda activate uGUIDE_env
```

You can then clone `μGUIDE` from the [Github repository](https://github.com/mjallais/uGUIDE):
```commandline
$ git clone git@github.com:mjallais/uGUIDE.git
```

Go into the installed repository and install it using `pip`:

```commandline
cd uGUIDE
pip install .
```

To test the installation worked fine, you can test:
```commandline
python -c "import uGUIDE"
```
It should not give any error message.