# Neural Networks From Scratch

This repository contains one NumPy feed-forward neural-network implementation and a notebook demonstrating it on the Parkinson's disease dataset.

## Project layout

```text
neural_network.py          neural-network implementation
parkinsons_demo.ipynb      training and evaluation demonstration
data/                      fixed Parkinson's train/test split
```

## Setup

```bash
python -m pip install -r requirements.txt
```

Open [`parkinsons_demo.ipynb`](parkinsons_demo.ipynb) and run all cells. The notebook loads the fixed split, standardizes features using training data only, trains the neural network, and reports test accuracy and F1 score.
