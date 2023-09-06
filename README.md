# 🧩 Self-Supervision is All You Need for Solving Rubik's Cube

[![arXiv](https://img.shields.io/badge/arXiv-2106.03157-b31b1b?style=for-the-badge)](https://arxiv.org/abs/2106.03157)
[![TMLR](https://img.shields.io/badge/TMLR_1188-112467?style=for-the-badge)](https://openreview.net/forum?id=bnBeNFB27b)
<!-- [![Try on Replicate](https://img.shields.io/badge/%F0%9F%9A%80%20Try%20on%20Replicate-black?style=for-the-badge)](https://replicate.com/kyo-takano/efficientcube) -->

This repository contains the code, models, and solutions as reported in the following paper:

K. Takano. Self-Supervision is All You Need for Solving Rubik's Cube. _Transactions on Machine Learning Research_, ISSN 2835-8856, 2023. URL: https://openreview.net/forum?id=bnBeNFB27b.

> **Abstract:**\
> Existing combinatorial search methods are often complex and require some level of expertise. This work introduces a simple and efficient deep learning method for solving combinatorial problems with a predefined goal, represented by Rubik's Cube. We demonstrate that, for such problems, training a deep neural network on random scrambles branching from the goal state is sufficient to achieve near-optimal solutions. When tested on Rubik's Cube, 15 Puzzle, and 7×7 Lights Out, our method outperformed the previous state-of-the-art method DeepCubeA, improving the trade-off between solution optimality and computational cost, despite significantly less training data. Furthermore, we investigate the scaling law of our Rubik's Cube solver with respect to model size and training data volume.

> [!IMPORTANT]
> The compute-optimal models trained using Half-Turn Metric (Section 7:  Scaling Law) are available in [**AlphaCube**](https://github.com/kyo-takano/alphacube), a dedicated Rubik's Cube solver based on this study.

## Code

### Jupyter Notebooks

We provide **_standalone_** Jupyter Notebooks for you to train DNNs & solve the problems we addressed in the paper.

| Problem | File | Launch |
| :--- | :--- | :--- |
| Rubik's Cube | [`./notebooks/Rubik's_Cube.ipynb`](http://github.com/kyo-takano/efficientcube/blob/main/notebooks/Rubik's_Cube.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyo-takano/efficientcube/blob/main/notebooks/Rubik's_Cube.ipynb) <br> [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/kyo-takano/efficientcube/blob/main/notebooks/Rubik's_Cube.ipynb) |
| 15 Puzzle | [`./notebooks/15_Puzzle.ipynb`](http://github.com/kyo-takano/efficientcube/blob/main/notebooks/15_Puzzle.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyo-takano/efficientcube/blob/main/notebooks/15_Puzzle.ipynb) <br> [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/kyo-takano/efficientcube/blob/main/notebooks/15_Puzzle.ipynb) |
| 7×7 Lights Out | [`./notebooks/Lights_Out.ipynb`](http://github.com/kyo-takano/efficientcube/blob/main/notebooks/Lights_Out.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyo-takano/efficientcube/blob/main/notebooks/Lights_Out.ipynb) <br> [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/kyo-takano/efficientcube/blob/main/notebooks/Lights_Out.ipynb) |

To fully replicate our study, you will need to modify hyperparameters.

### Package

We also provide a Python package at `./efficientcube` to solve a given problem. Please make sure that you have `torch>=1.12` installed.

**Installation:**

```bash
git clone https://github.com/kyo-takano/EfficientCube
```

**Example usage (Rubik's Cube):**

```python
from efficientcube import EfficientCube

""" Specify scramble & search parameter """
scramble = "D U F F L L U' B B F F D L L U R' F' D R' F' U L D' F' D R R"
beam_width = 1024 # This parameter controls the trade-off between speed and quality

""" Set up solver, apply scramble, & solve """
solver = EfficientCube(
    env ="Rubik's Cube",    # "Rubik's Cube", "15 Puzzle", or "Lights Out"
    model_path="auto",      # Automatically finds by `env` name
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

## Data 

### Models

Included in the package are the [TorchScript](https://pytorch.org/docs/stable/jit.html) models for the three problems. They can be specified as `./efficientcube/models/{cube3|puzzle15|lightsout7}.pth`. 

### Solutions

The solutions reported in the paper are located under `./results/`, and each problem has its own subdirectory containing Pickle file(s) (`./results/{cube3|puzzle15|lightsout7}/beam_width_{beam_width}.pkl`). Please see [`results/README.md`](http://github.com/kyo-takano/efficientcube/blob/main/results/README.md) for more details.

### Dataset

The DeepCubeA dataset we used for evaluation is available from either [Code Ocean](http://doi.org/10.24433/CO.4958495.v1) or [GitHub](http://github.com/forestagostinelli/DeepCubeA/).

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
