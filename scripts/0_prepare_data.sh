#! /bin/bash

cd ../

export DATASET_PATH=../Databases-and-code-for-l_p-functional-comparison/databases
export SAVE_PATH=../synolitic_data

uv run sgnn/obtain_data.py $DATASET_PATH $SAVE_PATH --data_size=1.0
uv run sgnn/preprocessing.py $SAVE_PATH --data_size=1.0

uv run sgnn/obtain_data.py $DATASET_PATH $SAVE_PATH --data_size=0.9
uv run sgnn/preprocessing.py $SAVE_PATH --data_size=0.9

uv run sgnn/obtain_data.py $DATASET_PATH $SAVE_PATH --data_size=0.7
uv run sgnn/preprocessing.py $SAVE_PATH --data_size=0.7

uv run sgnn/obtain_data.py $DATASET_PATH $SAVE_PATH --data_size=0.5
uv run sgnn/preprocessing.py $SAVE_PATH --data_size=0.5

uv run sgnn/obtain_data.py $DATASET_PATH $SAVE_PATH --data_size=0.4
uv run sgnn/preprocessing.py $SAVE_PATH --data_size=0.4

uv run sgnn/obtain_data.py $DATASET_PATH $SAVE_PATH --data_size=0.2
uv run sgnn/preprocessing.py $SAVE_PATH --data_size=0.2

uv run sgnn/obtain_data.py $DATASET_PATH $SAVE_PATH --data_size=0.1
uv run sgnn/preprocessing.py $SAVE_PATH --data_size=0.1

uv run sgnn/obtain_data.py $DATASET_PATH $SAVE_PATH --data_size=0.05
uv run sgnn/preprocessing.py $SAVE_PATH --data_size=0.05
