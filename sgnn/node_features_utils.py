import networkx as nx
import torch
from joblib import Parallel, delayed
from tqdm import tqdm


def _process_single_graph(graph):
    """Auxiliary function to process a single graph."""
    if torch.isnan(graph.x).any():
        raise ValueError("NaN detected in node features for a graph.")

    graph = graph.clone()
    edge_index = graph.edge_index.cpu().numpy()
    edge_weights = graph.edge_attr.squeeze().cpu().numpy()
    num_nodes = graph.num_nodes

    G = nx.Graph()
    for (i, j), w in zip(edge_index.T, edge_weights, strict=False):
        if w != 0:
            G.add_edge(int(i), int(j), weight=float(w))
    G.add_nodes_from(range(num_nodes))

    deg = dict(G.degree())
    strength = dict(G.degree(weight="weight"))
    closeness = nx.closeness_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight="weight")

    scalar_features = (
        graph.x.squeeze()
        if graph.x.dim() == 2 and graph.x.size(1) == 1
        else torch.zeros(num_nodes)
    )

    node_features = []
    for node in range(num_nodes):
        scalar = scalar_features[node].item()
        features = [
            scalar,
            deg.get(node, 0) / max(num_nodes - 1, 1),
            strength.get(node, 0.0) / max(num_nodes - 1, 1),
            closeness.get(node, 0.0),
            betweenness.get(node, 0.0),
        ]
        node_features.append(features)

    graph.x = torch.tensor(node_features, dtype=torch.float)

    if torch.isnan(graph.x).any():
        graph.x = torch.nan_to_num(graph.x)

    return graph


def add_node_features(graphs):
    graph_data = {}
    for data_type in ["train", "test"]:
        print("Adding node features for", data_type)
        new_graphs = Parallel(n_jobs=8, verbose=0, prefer="processes")(
            delayed(_process_single_graph)(graph)
            for graph in tqdm(graphs[data_type])
        )
        graph_data[data_type] = new_graphs

    return graph_data
