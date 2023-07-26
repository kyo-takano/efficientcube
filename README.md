# 🧩 Self-Supervision is All You Need for Solving Rubik's Cube

[![arXiv](https://img.shields.io/badge/arXiv-2106.03157-b31b1b?style=for-the-badge)](https://arxiv.org/abs/2106.03157)
[![TMLR](https://img.shields.io/badge/TMLR_1188-112467?style=for-the-badge)](https://openreview.net/forum?id=bnBeNFB27b)
<!-- [![Try on Replicate](https://img.shields.io/badge/%F0%9F%9A%80%20Try%20on%20Replicate-black?style=for-the-badge)](https://replicate.com/kyo-takano/efficientcube) -->

This repository contains the source code, models, and solutions as reported in the following paper:

K. Takano. Self-Supervision is All You Need for Solving Rubik's Cube. _Transactions on Machine Learning Research_, ISSN 2835-8856, 2023. \
URL: https://openreview.net/forum?id=bnBeNFB27b.

> **Abstract:**\
> Existing combinatorial search methods are often complex and require some level of expertise. This work introduces a simple and efficient deep learning method for solving combinatorial problems with a predefined goal, represented by Rubik's Cube. We demonstrate that, for such problems, training a deep neural network on random scrambles branching from the goal state is sufficient to achieve near-optimal solutions. When tested on Rubik's Cube, 15 Puzzle, and 7×7 Lights Out, our method outperformed the previous state-of-the-art method DeepCubeA, improving the trade-off between solution optimality and computational cost, despite significantly less training data. Furthermore, we investigate the scaling law of our Rubik's Cube solver with respect to model size and training data volume.

## Code

We provide **_standalone_** Jupyter Notebooks for you to train DNNs & solve problems we addressed in the paper.

| Problem | File | Launch |
| --- | --- | --- |
| Rubik's Cube | [`./notebooks/Rubik's_Cube.ipynb`](./notebooks/Rubik's_Cube.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyo-takano/efficientcube/blob/main/notebooks/Rubik's_Cube.ipynb) <br> [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/kyo-takano/efficientcube/blob/main/notebooks/Rubik's_Cube.ipynb) |
| 15 Puzzle | [`./notebooks/15_Puzzle.ipynb`](./notebooks/15_Puzzle.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyo-takano/efficientcube/blob/main/notebooks/15_Puzzle.ipynb) <br> [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/kyo-takano/efficientcube/blob/main/notebooks/15_Puzzle.ipynb) |
| 7×7 Lights Out | [`./notebooks/Lights_Out.ipynb`](./notebooks/Lights_Out.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyo-takano/efficientcube/blob/main/notebooks/Lights_Out.ipynb) <br> [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/kyo-takano/efficientcube/blob/main/notebooks/Lights_Out.ipynb) |

To fully replicate our study, you will need to modify hyperparameters.

## Models

We release [TorchScript](https://pytorch.org/docs/stable/jit.html) models for the three problems, which can be specified as `./efficientcube/models/{cube3|puzzle15|lightsout7}/{training_steps}steps_scripted.pth`. 
To use these models, please ensure that you have `torch>=1.12` installed.

## Solutions

The solutions reported in the paper are located under `./results/`, and each problem has its own subdirectory containing Pickle file(s) (`./results/{cube3|puzzle15|lightsout7}/n={training_steps}.k={beam_width}.pkl`).

Each of the files holds a set of results (`solutions`, `times`, and `num_nodes_generated`) obtained with parameters `n` (number of training steps) and `k` (beam width).
Please note that we only include the _actual_ times taken per solution.

```python
import pickle

with open("./results/cube3/n=2000000.k=262144.pkl", "rb") as f:
    data = pickle.load(f)
    solutions, times, num_nodes_generated = [data[k] for k in ['solutions', 'times', 'num_nodes_generated']]

mean_solution_length = sum([len(e) for e in solutions]) / len(solutions)
```

The DeepCubeA dataset is available from either [Code Ocean](http://doi.org/10.24433/CO.4958495.v1) or [GitHub](http://github.com/forestagostinelli/DeepCubeA/).

## Package

This repository also contains a Python package at `./efficientcube` to scramble & solve problems. 
As it depends on TorchScript, to use the package, you need to have `torch>=1.12` installed.

Example usage (Rubik's Cube):

```python
from efficientcube import EfficientCube

""" Specify scramble & search parameter """
scramble = "D U F F L L U' B B F F D L L U R' F' D R' F' U L D' F' D R R"
beam_width = 1024 # This parameter controls the trade-off between speed and quality

""" Set up solver, apply scramble, & solve """
solver = EfficientCube(
    env ="cube3",       # "cube3", "puzzle15", or "lightsout7"
    model_path="auto",  # Automatically finds by `env` name
)
solver.apply_moves_to_env(scramble)
result = solver.solve(beam_width)

""" Verify the result """
if result is not None:
    print('Solution:', ' '.join(result['solutions']))
    print('Length:', len(result['solutions']))
    solver.reset_env()
    solver.apply_moves_to_env(scramble.split() + result['solutions'])
    assert solver.env_is_solved()
else:
    print('Failed')
```

## Citation

```bibtex
@article{
    takano2023selfsupervision,
    title={Self-Supervision is All You Need for Solving Rubik's Cube},
    author={Kyo Takano},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2023},
    url={https://openreview.net/forum?id=bnBeNFB27b}
}
```

## License

All materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY-4.0).
You may obtain a copy of the license at: https://creativecommons.org/licenses/by/4.0/legalcode

## Contact

Please contact the author at <code><a href="mailto:kyo.takano@mentalese.co" target="_blank">kyo.takano@mentalese.co</a></code> for any questions.
