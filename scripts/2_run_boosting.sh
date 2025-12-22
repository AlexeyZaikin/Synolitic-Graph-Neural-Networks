#! /bin/bash

cd ../

export DATA_DIR=../synolitic_data

for fold in {0..4}; do
    uv run sgnn/train_xgboost.py ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=0.05 ++data.fold=$fold
    uv run sgnn/train_xgboost.py ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=0.1 ++data.fold=$fold
    uv run sgnn/train_xgboost.py ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=0.2 ++data.fold=$fold
    uv run sgnn/train_xgboost.py ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=0.4 ++data.fold=$fold
    uv run sgnn/train_xgboost.py ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=0.5 ++data.fold=$fold
    uv run sgnn/train_xgboost.py ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=0.7 ++data.fold=$fold
    uv run sgnn/train_xgboost.py ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=0.9 ++data.fold=$fold
    uv run sgnn/train_xgboost.py ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=1.0 ++data.fold=$fold
done
