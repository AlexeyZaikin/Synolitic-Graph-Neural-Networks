#! /bin/bash

cd ../

export DATA_DIR=../synolitic_data

for fold in {0..4}; do
    CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GATv2 ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=0.05 ++data.fold=$fold    
    CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GATv2 ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=0.1 ++data.fold=$fold
    CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GATv2 ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=0.2 ++data.fold=$fold
    CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GATv2 ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=0.4 ++data.fold=$fold
    CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GATv2 ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=0.5 ++data.fold=$fold
    CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GATv2 ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=0.7 ++data.fold=$fold
    CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GATv2 ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=0.9 ++data.fold=$fold
    CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GATv2 ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=1.0 ++data.fold=$fold
done

for fold in {0..4}; do
    CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GCN ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=0.05 ++data.fold=$fold
    CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GCN ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=0.1 ++data.fold=$fold
    CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GCN ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=0.2 ++data.fold=$fold
    CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GCN ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=0.4 ++data.fold=$fold
    CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GCN ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=0.5 ++data.fold=$fold
    CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GCN ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=0.7 ++data.fold=$fold
    CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GCN ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=0.9 ++data.fold=$fold
    CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GCN ++data.dataset_path=$DATA_DIR ++per_dataset=False ++data.dataset_size=1.0 ++data.fold=$fold
done