# GNN Models Project Overview

This project consists of various scripts designed for data processing, model configuration, and hyperparameter optimization specific to Graph Neural Networks (GNNs). The primary goal is to evaluate the performance of different GNN models under conditions with missing and inaccurate labels. The following sections provide detailed descriptions of each file, how to run them, and the dependencies required.

## Files Description

### 1. `combine.py`
This script is used for running and obtaining classification results for three different GNN models: GAT (Graph Attention Network), GCN (Graph Convolutional Network), and GraphSAGE. It evaluates the models' performance with missing and inaccurate labels. Before running this script, ensure that you generate the dataset, preprocess it, and apply noise to the datasets. Also, set the correct base path or folder for all datasets.

### 2. `modules.py`
This script contains various helper functions and modules used throughout the project, such as data processing functions and utility functions.

### 3. `data.py`
This script handles data loading and transformation. It includes functions to prepare data for model training and evaluation.

### 4. `model_configs.py`
This script defines various configurations and parameters for different GNN models.

### 5. `combine_optuna.py`
This script integrates the Optuna library for hyperparameter optimization. It tunes the hyperparameters of the GNN models to achieve optimal performance.

