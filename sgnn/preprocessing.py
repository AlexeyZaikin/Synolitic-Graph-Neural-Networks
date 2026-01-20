import argparse
import pickle  # noqa: S403
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


def prepare_tabular(args):
    # Determine base path based on noisy flag
    if args.noisy:
        # For noisy data, check if we have fold structure in noisy directory
        base_path = Path(args.data_path) / f"csv_{args.data_size}" / "noisy"
    else:
        base_path = Path(args.data_path) / f"csv_{args.data_size}"
    
    # Check if fold directories exist (for all data sizes)
    fold_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    
    if fold_dirs:
        # Process each fold directory
        for fold_dir in tqdm(fold_dirs, desc="Processing folds"):
            _process_single_directory(fold_dir, fold_dir)
    else:
        # No folds found, process as before (backward compatibility)
        _process_single_directory(base_path, base_path)


def _process_single_directory(data_dir: Path, output_dir: Path):
    """Process graph data from a single directory and save processed_graphs.pkl"""
    datasets = set()
    datasets.update(
        p_i.stem.split(".")[0]
        for p_i in data_dir.glob("*.graph.csv")
    )

    all_data = {}
    for dataset in tqdm(datasets, desc="Graph structure building", leave=False):
        all_data[dataset] = defaultdict(list)
        graph_data = pd.read_csv(data_dir / f"{dataset}.graph.csv")
        node_features_data = pd.read_csv(
            data_dir / f"{dataset}.node_features.csv",
            index_col=0,
        )
        # Get the graph columns (exclude p1, p2, feature columns, and is_in_test)
        graph_columns = []
        for col in graph_data.columns:
            if col not in {"p1", "p2", "is_in_test"} and not col.startswith("num__feature_"):
                graph_columns.append(col)

        n_graphs = len(graph_columns)
        for k in range(n_graphs):
            x = []
            y = []
            edge_index = []
            edge_attr = []
            for r in range(len(graph_data) - 1):
                i = int(graph_data["p1"].iloc[r].split("_")[-1])
                j = int(graph_data["p2"].iloc[r].split("_")[-1])
                v = graph_data[graph_columns[k]].iloc[r]
                edge_index.append((i, j))
                edge_attr.append((v,))
            x.append(node_features_data.iloc[k].to_numpy()[:-1])
            y = bool(node_features_data.iloc[k].to_numpy()[-1])
            is_test = bool(graph_data.iloc[len(graph_data) - 1][graph_columns[k]])
            data = Data(
                x=torch.Tensor(x).T,
                edge_index=torch.LongTensor(edge_index).T,
                edge_attr=torch.Tensor(edge_attr),
                y=y,
                dataset_name=dataset,
            )
            if is_test:
                all_data[dataset]["test"].append(data)
            else:
                all_data[dataset]["train"].append(data)

    with (output_dir / "processed_graphs.pkl").open("wb") as f:
        pickle.dump(all_data, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("--data_size", type=float, default=1.0)
    parser.add_argument("--noisy", action="store_true")
    args = parser.parse_args()
    prepare_tabular(args)


if __name__ == "__main__":
    main()
