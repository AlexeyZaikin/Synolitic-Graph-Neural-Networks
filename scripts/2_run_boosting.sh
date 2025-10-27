#! /bin/bash

cd ../

export DATA_DIR=../synolitic_data

uv run sgnn/train_xgboost.py ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=0.05
uv run sgnn/train_xgboost.py ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=0.1
uv run sgnn/train_xgboost.py ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=0.2
uv run sgnn/train_xgboost.py ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=0.4
uv run sgnn/train_xgboost.py ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=0.5
uv run sgnn/train_xgboost.py ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=0.7
uv run sgnn/train_xgboost.py ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=0.9
uv run sgnn/train_xgboost.py ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=1.0
