# 🚧 Sharp Star ✨ 🚧

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-%3E=3.10-blue.svg)](https://www.python.org/downloads/)
![Tests](https://github.com/ArisChristoforidis/sharp-star/actions/workflows/tests.yaml/badge.svg?style=social)
![Last Commit](https://img.shields.io/github/last-commit/ArisChristoforidis/sharp-star)
![Open Issues](https://img.shields.io/github/issues/ArisChristoforidis/sharp-star)
![Framework](https://img.shields.io/badge/framework-PyTorch-red)
![Model](https://img.shields.io/badge/model-UNet-blue)
![Status](https://img.shields.io/badge/status-training-informational)

### Sharp Star is a neural model that sharpens and deblurs your astro images.

## Table of Contents

* [Installation](#installation)
* [Usage](#usage)
* [Dataset](#dataset)
* [Model](#model)
* [Evaluation](#evaluation)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

## Installation

To get started, clone the repository:

```bash
git clone https://github.com/ArisChristoforidis/sharp-star
cd sharp-star
```

[__Recommended__] Create a virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS and Linux
pip install -r requirements.txt
```

## Usage

### Inference

Run `predict.py` to perform inference on an image.

__Options__


`--input`, `-i`:  Path to the input image.

`--output`, `-o`: Path to save the output (predicted) image.

`--model`, `-m`: Path to the trained model file.

`--batch`, `-b`: Batch size for processing the image.

`--patch`, `-p`: Size of the image patches used for processing.

__Example__

```
python predict.py --input input.jpg --output out.jpg --model model.pth --batch 8 --patch 512
```

### Training

You can use `train.py` to train your own model.

__Options__

`--checkpoint`, `-c`:  The path to the model checkpoint if it exists.

`--train`, `-t`: The path to the train set.

`--eval`, `-v`: The path to the evaluation set.

`--output`, `-o`: The output path for the model.

`--learning_rate`, `-lr`: The default learning rate.

`--batch`, `-b`: The batch size.

`--epochs`, `-e`: The number of epochs to train for.

`--log`, `-l`: Whether to log metrics to wandb or not.

__Example__

```
python train.py --checkpoint models/model.pth --train data/train --eval data/eval output models/updated_model.pth --learning_rate 0.001 --batch 32 --epochs 20 --log True
```

# Common commands

- `coverage run -m pytest`: Run the tests to calculate code coverage
- `coverage report -m`: Show coverage stats from last test run.
- `ruff check . --fix`: Runs the linter and fixes some things, e.g. the import order.
- `ruff format <file>`: Formats a single file.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
