# Counter Strike 2 (CS2) Behavioral Latent Transfer Learning

This repository contains the code and data for modeling individual behavioral "signatures" in CS2 trajectory data and transferring them across different domains (maps and roles).

## Overview

The core of this project is a **Multi-Source Multi-Target Transfer Learning** framework. It extracts individual-specific traits into a latent space ($z$) from multiple source environments and uses a HyperNetwork to adapt a movement predictor to new target environments.

### Key Components:
- **Encoder**: Maps long-term trajectory history (60s) to a latent vector.
- **Combiner**: Aggregates latent vectors from various available source domains.
- **HyperNetwork**: Generates the weights for a target predictor based on the aggregated latent $z$.
- **Target Predictor**: A lightweight model that predicts future positions based on short-term history.

## Repository Structure

- `data/`: Processed trajectory sequences in JSON format.
- `models.py`: Neural network architectures (PyTorch).
- `utils.py`: Data loading, preprocessing, and utility functions.
- `train.py`: Script to train the transfer model.
- `evaluate.py`: Script to evaluate the trained model on test data.
- `analysis_zdim.py`: Tool to analyze the latent space and dimensionality.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas, Matplotlib, Scipy, TQDM, Scikit-learn

### Installation
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Training
To train the multi-transfer model:
```bash
python train.py --epochs 1000 --z_dim 8 --model_dir checkpoints
```

#### 2. Evaluation
To evaluate the best model against baselines:
```bash
python evaluate.py --model_path checkpoints/best_model.pth --z_dim 8
```

#### 3. Latent Analysis
To visualize the learned latent space:
```bash
python analysis_zdim.py --model_dir checkpoints --z_dims 2 4 8
```

## Data Format
The JSON files in `data/` contain sequences of (X, Y) coordinates for a player and the center of their allies.
- `X, Y`: Player coordinates.
- `Ally_X, Ally_Y`: Average coordinates of teammate players.

## Citation
If you use this code or data in your research, please cite our work (details to be added).
