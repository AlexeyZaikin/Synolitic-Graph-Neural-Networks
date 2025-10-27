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
    datasets = set()
    datasets.update(
        str(p).split("/")[-1].split(".")[0]
        for p in (Path(args.data_path) / (f"csv_{args.data_size}" if not args.noisy else "")).glob(
            "*.csv"
        )
    )

    all_data = {}
    for dataset in tqdm(datasets, desc="Graph structure building"):
        all_data[dataset] = defaultdict(list)
        graph_data = pd.read_csv(
            Path(args.data_path)
            / (f"csv_{args.data_size}" if not args.noisy else "")
            / f"{dataset}.graph.csv",
        )
        node_features_data = pd.read_csv(
            Path(args.data_path)
            / (f"csv_{args.data_size}" if not args.noisy else "")
            / f"{dataset}.node_features.csv",
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

    with (
        Path(args.data_path)
        / (f"csv_{args.data_size}" if not args.noisy else "")
        / "processed_graphs.pkl"
    ).open("wb") as f:
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
