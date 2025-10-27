# Overcoming the Curse of Dimensionality with Synolitic AI

This repository contains the implementation of experiments from the paper "Overcoming the Curse of Dimensionality with Synolitic AI". It provides a comprehensive pipeline for conducting experiments with synolitic graphs using Graph Neural Networks (GNNs) and XGBoost for classification tasks.

## Overview

This project implements a comprehensive machine learning pipeline that:

- Converts [tabular data](https://github.com/Mirkes/Databases-and-code-for-l_p-functional-comparison) into synolitic graphs
- Trains Graph Neural Networks ([GATv2](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.GATv2Conv.html), [GCN](https://pytorch-geometric.readthedocs.io/en/2.5.2/generated/torch_geometric.nn.conv.GCNConv.html)) on graph representations
- Trains XGBoost baseline models on tabular features
- Supports feature expansion with noisy features for robustness testing
- Provides comprehensive hyperparameter optimization using Optuna

## Features

- **Multiple GNN Architectures**: GATv2, GCN
- **Graph Sparsification**: Various sparsification strategies (percentage-based, minimum connected, no sparsification)
- **Node Features**: Optional addition of node features to graphs (degree centrality, clustering coefficient, betweenness centrality, eigenvector centrality, PageRank, local clustering coefficient)
- **Noisy Features**: Feature expansion with controlled noise injection for robustness evaluation
- **Hyperparameter Optimization**: Automated hyperparameter tuning using Optuna
- **Cross-validation**: K-fold cross-validation support
- **Leave-One-Out Validation**: Leave-one-dataset-out cross-validation for multi-dataset experiments
- **Comprehensive Logging**: TensorBoard integration and detailed performance metrics tracking

## Installation

### Prerequisites

- Python 3.13+
- CUDA-compatible GPU (for GNN training)
- UV package manager

### Setup

1. Clone this repository and go to the `smiles_2025_sgnn` directory

2. Install dependencies using UV:

   ```bash
   uv sync
   ```

3. Clone [tabular data](https://github.com/Mirkes/Databases-and-code-for-l_p-functional-comparison)

4. Set up environment variables:

   ```bash
   export DATASET_PATH=/path/to/your/databases
   export SAVE_PATH=/path/to/save/results
   export DATA_DIR=/path/to/processed/data
   ```

## Usage

The pipeline consists of four main steps in the `scripts` directory:

### 1. Data Preparation

```bash
./0_prepare_data.sh
```

- Converts .mat files to CSV format with graph and node feature representations
- Creates synolitic graphs for different dataset proportions (1.0, 0.9, 0.7, 0.5, 0.4, 0.2, 0.1, 0.05)

### 2. Graph Neural Network Training

```bash
./1_run_graphs_training.sh
```

- Trains GATv2 and GCN models on different dataset proportions
- Supports hyperparameter optimization with Optuna
- Generates comprehensive training logs and performance metrics

### 3. XGBoost Training

```bash
./2_run_boosting.sh
```

- Trains XGBoost models on tabular features
- Supports grid search for hyperparameter tuning
- Evaluates model performance across different dataset sizes

### 4. Feature Expansion

```bash
./3_expand_features.sh
```

- Adds noisy features to test model robustness
- Creates duplicate features with controlled noise injection (5% noise level)
- Useful for analyzing model sensitivity to feature noise and evaluating robustness

## Configuration

Modify `conf/config.yaml` to customize:

- Model architectures and hyperparameters
- Training parameters (learning rate, batch size, number of epochs)
- Data processing options (sparsification, node features)
- Optimization settings (number of Optuna trials, timeout)

### Experiment Modes

The pipeline supports three different experiment modes:

1. **Combined Training** (default): All datasets are combined into a single training and test set (Foundation model task)

   - Set: `leave_one_out: False` and `per_dataset: False`

2. **Per-Dataset Training**: Each dataset is trained and evaluated separately (Separate datasets task)

   - Set: `per_dataset: True`
   - Results are saved in separate directories for each dataset

3. **Leave-One-Out Validation**: Leave-one-dataset-out cross-validation
   - Set: `leave_one_out: True`
   - For each iteration, one dataset is held out for testing while all other datasets are combined for training
   - This approach evaluates model generalization across different datasets
   - Results are saved in directories named `leave_one_out_{dataset_name}`

## Advanced Usage

### Training Individual Models

```bash
# Train a specific GNN configuration
uv run main.py ++model.type=GATv2 ++data.dataset_size=0.5

# Train with hyperparameter optimization
uv run main.py ++optimize=True

# Train with noisy features
uv run main.py ++expand_features=True

# Train with leave-one-out validation
uv run main.py ++leave_one_out=True

# Train on each dataset separately
uv run main.py ++per_dataset=True
```

### XGBoost Training

```bash
# Train XGBoost with grid search
uv run sgnn/train_xgboost.py ++xgboost.gridsearch.enabled=true
```

## Project Structure

```text
smiles_2025_sgnn/
├── conf/
│   └── config.yaml          # Main configuration file
├── scripts/
│   ├── 0_prepare_data.sh    # Data preparation script
│   ├── 1_run_graphs_training.sh  # GNN training script
│   ├── 2_run_boosting.sh    # XGBoost training script
│   └── 3_expand_features.sh # Feature expansion script
├── sgnn/
│   ├── model.py             # GNN model implementations
│   ├── trainer.py           # Training utilities
│   ├── obtain_data.py       # Data conversion utilities
│   ├── expand_features.py   # Noisy feature generation
│   ├── preprocessing.py     # Data preprocessing
│   ├── sparsify_utils.py    # Graph sparsification
│   ├── node_features_utils.py # Node feature addition
│   ├── train_xgboost.py     # XGBoost training
│   └── utils.py             # General utilities
├── main.py                  # Main training script
└── analyze_results.ipynb    # Results analysis notebook
```

## Results and Logging

- Training metrics are logged to TensorBoard for visualization
- Model checkpoints are automatically saved during training
- Comprehensive evaluation metrics (accuracy, F1-score, AUC-ROC)
- Experimental results can be analyzed using the provided Jupyter notebook
