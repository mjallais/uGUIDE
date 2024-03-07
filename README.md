# μGUIDE

[Documentation](https://mjallais.github.io/uGUIDE/)

This is the official implementation of ``μGUIDE`` presented in https://arxiv.org/abs/2312.17293. To cite it, please use:
```bibtex
@misc{jallais2023muguide,
      title={$\mu$GUIDE: a framework for microstructure imaging via generalized uncertainty-driven inference using deep learning}, 
      author={Maëliss Jallais and Marco Palombo},
      year={2023},
      eprint={2312.17293},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

``μGUIDE`` is a Python library for efficiently estimating posterior distributions of microstructure parameters from diffusion MRI signals.

## Installation

To get started, install ``μGUIDE`` on your machine via pip:

1. We recommend to use a  [`conda`](https://docs.conda.io/en/latest/miniconda.html) virtual environment. If `conda` is installed on your machine, an environment for installing ``μGUIDE`` can be created as follows:

```shell
conda create -n uGUIDE_env python=3.8 && conda activate uGUIDE_env
```

2. Fork the repository and run the following command to clone it on your local machine:

```shell
git clone git@github.com:{YOUR_GITHUB_USERNAME}/uGUIDE.git
```

3. ``cd`` to ``μGUIDE`` directory and install it:

```shell
cd uGUIDE
pip install .
```
This will also install the dependencies of ``μGUIDE``.

4. To check if the installation worked fine, you can do:

```shell
python -c 'import uGUIDE'
```

and it should not give any error message.

Visit [``μGUIDE`` documentation](https://mjallais.github.io/uGUIDE/) for more information.

## Cite

If you use ``μGUIDE``, please cite:
```bibtex
@misc{jallais2023muguide,
      title={$\mu$GUIDE: a framework for microstructure imaging via generalized uncertainty-driven inference using deep learning}, 
      author={Maëliss Jallais and Marco Palombo},
      year={2023},
      eprint={2312.17293},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

## Further links
- https://arxiv.org/abs/2312.17293
