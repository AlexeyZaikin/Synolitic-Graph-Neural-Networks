#! /bin/bash
CUDA_VISIBLE_DEVICES=0 uv run main.py ++model.type=GATv2
CUDA_VISIBLE_DEVICES=1 uv run main.py ++model.type=GCN
CUDA_VISIBLE_DEVICES=2 uv run main.py ++model.type=GATv2 ++data.dataset_path=data/hypergraph_tabular/g3
CUDA_VISIBLE_DEVICES=3 uv run main.py ++model.type=GATv2 ++data.dataset_path=data/hypergraph_tabular/g4
CUDA_VISIBLE_DEVICES=4 uv run main.py ++model.type=GATv2 ++data.dataset_path=data/hypergraph_tabular/g5
CUDA_VISIBLE_DEVICES=5 uv run main.py ++model.type=GCN ++data.dataset_path=data/hypergraph_tabular/g3
CUDA_VISIBLE_DEVICES=6 uv run main.py ++model.type=GCN ++data.dataset_path=data/hypergraph_tabular/g4
CUDA_VISIBLE_DEVICES=7 uv run main.py ++model.type=GCN ++data.dataset_path=data/hypergraph_tabular/g5
