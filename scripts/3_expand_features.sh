#! /bin/bash

cd ../

export DATA_DIR=../synolitic_data

uv run sgnn/expand_features.py $DATA_DIR
uv run sgnn/preprocessing.py $DATA_DIR/csv_1.0/noisy --noisy
