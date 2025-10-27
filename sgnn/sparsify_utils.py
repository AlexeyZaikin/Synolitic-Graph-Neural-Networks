from collections.abc import Callable
from functools import partial

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


def no_sparsify(graphs: dict[str, list[Data]]):
    """Return graphs without sparsification (cloned for consistency)."""
    graph_data = {}
    for data_type in ["train", "test"]:
        graph_data[data_type] = [graph.clone() for graph in graphs[data_type]]
    return graph_data


def sparsify_p(graphs: dict[str, list[Data]], p: float):
    """p — fraction of edges with highest absolute weights."""
    graph_data = {}
    for data_type in ["train", "test"]:
        new_graphs = []

        for ograph in graphs[data_type]:
            graph = ograph.clone()
            if graph.edge_attr is None or graph.edge_index is None:
                raise ValueError("Incorrect graph as there is no edge_attr or edge_index")

            edge_attr = graph.edge_attr.squeeze()

            num_edges_to_keep = int(np.ceil(len(edge_attr) * p))
            edge_attr = (edge_attr - 0.5).abs()
            threshold = torch.topk(edge_attr, num_edges_to_keep, largest=True).values.min()
            mask = edge_attr >= threshold

            graph.edge_index = graph.edge_index[:, mask]
            graph.edge_attr = graph.edge_attr[mask, :]

            new_graphs.append(graph)

        graph_data[data_type] = new_graphs

    return graph_data


def sparsify_min_connected(graphs: dict[str, list[Data]]):
    graph_data = {}
    for data_type in ["train", "test"]:
        new_graphs = []

        for ograph in graphs[data_type]:
            graph = ograph.clone()
            if graph.edge_attr is None or graph.edge_index is None:
                raise ValueError("Incorrect graph as there is no edge_attr or edge_index")

            edge_index = graph.edge_index
            edge_attr = graph.edge_attr.squeeze()
            edge_attr -= 0.5
            abs_weights = edge_attr.abs()
            unique_weights = torch.unique(abs_weights)
            unique_weights, _ = torch.sort(unique_weights)
            low = 0
            high = len(unique_weights) - 1
            best_eps = unique_weights[-1]

            # Binary search over unique weights
            while low <= high:
                mid = (low + high) // 2
                eps = unique_weights[mid]

                # Build graph with edges where abs(w) >= eps
                mask = abs_weights >= eps
                edges = edge_index[:, mask].T.cpu().numpy()
                G = nx.Graph()
                G.add_edges_from(edges)
                G.add_nodes_from(range(graph.num_nodes or 1))

                if nx.is_connected(G):
                    best_eps = eps
                    low = mid + 1
                else:
                    high = mid - 1

            # Remove weak edges
            mask = abs_weights >= best_eps

            graph.edge_attr = graph.edge_attr[mask, :]
            graph.edge_index = graph.edge_index[:, mask]
            new_graphs.append(graph)

        graph_data[data_type] = new_graphs

    return graph_data


def get_sparsify_f_list(
    p_list: list[float] | None = None,
) -> list[tuple[str, Callable[..., dict]]]:
    if p_list is None:
        p_list = [0.4, 0.6, 0.8]
    return (
        [("no_sparsify", no_sparsify)]
        + [
            (f"sparsify_p_{p_val}".replace(".", "_"), partial(sparsify_p, p=p_val))
            for p_val in p_list
        ]
        + [("sparsify_min_connected", sparsify_min_connected)]
    )
