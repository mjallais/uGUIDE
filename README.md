# μGUIDE

[Documentation](https://mjallais.github.io/uGUIDE/)

This is the official implementation of ``μGUIDE`` presented in [eLife](https://elifesciences.org/articles/101069). To cite it, please use:
```bibtex
@article{10.7554/eLife.101069,
      article_type = {journal},
      title = {Introducing µGUIDE for quantitative imaging via generalized uncertainty-driven inference using deep learning},
      author = {Jallais, Maëliss and Palombo, Marco},
      editor = {Sui, Jing and Walczak, Aleksandra M},
      volume = 13,
      year = 2024,
      month = {nov},
      pub_date = {2024-11-26},
      pages = {RP101069},
      citation = {eLife 2024;13:RP101069},
      doi = {10.7554/eLife.101069},
      url = {https://doi.org/10.7554/eLife.101069},
      journal = {eLife},
      issn = {2050-084X},
      publisher = {eLife Sciences Publications, Ltd},
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
@article{10.7554/eLife.101069,
      article_type = {journal},
      title = {Introducing µGUIDE for quantitative imaging via generalized uncertainty-driven inference using deep learning},
      author = {Jallais, Maëliss and Palombo, Marco},
      editor = {Sui, Jing and Walczak, Aleksandra M},
      volume = 13,
      year = 2024,
      month = {nov},
      pub_date = {2024-11-26},
      pages = {RP101069},
      citation = {eLife 2024;13:RP101069},
      doi = {10.7554/eLife.101069},
      url = {https://doi.org/10.7554/eLife.101069},
      journal = {eLife},
      issn = {2050-084X},
      publisher = {eLife Sciences Publications, Ltd},
}
```

## Further links
- https://elifesciences.org/articles/101069
