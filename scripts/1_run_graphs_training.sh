#! /bin/bash

cd ../

export DATA_DIR=../synolitic_data

CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GATv2 ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=0.05
CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GATv2 ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=0.1
CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GATv2 ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=0.2
CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GATv2 ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=0.4
CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GATv2 ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=0.5
CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GATv2 ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=0.7
CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GATv2 ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=0.9
CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GATv2 ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=1.0

CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GCN ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=0.05
CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GCN ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=0.1
CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GCN ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=0.2
CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GCN ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=0.4
CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GCN ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=0.5
CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GCN ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=0.7
CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GCN ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=0.9
CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GCN ++data.dataset_path=$DATA_DIR ++per_dataset=True ++data.dataset_size=1.0
