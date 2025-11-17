import torch
import logging
import os
import numpy as np
import sklearn.metrics.pairwise as smp
import hdbscan
import math
import networkx as nx
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import pdb

# --- Utility Functions (unchanged) ---
def get_update_norm(local_update):
    squared_sum = 0
    for name, value in local_update.items():
        if 'tracked' in name or 'running' in name:
            continue
        squared_sum += torch.sum(torch.pow(value, 2)).item()
    update_norm = math.sqrt(squared_sum)
    return update_norm

# --- Optimized Visualization Function ---
def visualize_fedgraph(graph: nx.Graph, labels: list, scores: np.ndarray, round_idx: int, output_dir: str, clients_this_round: list, node_features: list):
    """
    Visualizes the FedGraph for a given round with enhanced aesthetics and information.

    Parameters:
    - graph: networkx graph object
    - labels: HDBSCAN cluster labels
    - scores: HDBSCAN membership scores (probabilities_)
    - round_idx: current training round
    - output_dir: directory to save the image
    - clients_this_round: list of client IDs for the current round
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Node color and size mapping
    node_colors = []
    node_sizes = []
    
    for i, client_id in enumerate(clients_this_round):
        score = scores[i]
        
        if labels[i] == -1:
            node_colors.append('red')  # Outliers are red
            node_sizes.append(150)  # Fixed size for outliers
        else:
            # Use a color map for a smoother gradient based on score
            # The score is normalized to the [0.2, 1.0] range for better visual distinction
            normalized_score = score * 0.8 + 0.2 
            node_colors.append(plt.cm.Greens(normalized_score))
            node_sizes.append(250 + score * 400) # Size increases with score
    
    pos = nx.spring_layout(graph, seed=42, k=0.3)  # Use force-directed layout

    # Draw nodes and edges
    edge_attrs = nx.get_edge_attributes(graph, 'weight')
    if edge_attrs:
        # Draw edges
        edges, weights = zip(*edge_attrs.items())
        edge_weights_scaled = [w * 7 for w in weights] # Scale weights for better visibility
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=edges, width=edge_weights_scaled, edge_color='gray', alpha=0.6)
        
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.9, linewidths=1.5, edgecolors='black')
    
    # Add labels with client ID
    node_labels = {i: f"C{client_id}\nScore: {scores[i]:.2f}" for i, client_id in enumerate(clients_this_round)}
    nx.draw_networkx_labels(graph, pos, labels=node_labels, ax=ax, font_size=10, font_family='sans-serif', font_weight='bold')

    # Add a title and legend for clarity
    ax.set_title(f"FedGraph Topology - Round {round_idx}", fontsize=20, fontweight='bold')
    
    # Create legend handles
    legend_handles = [
        mpatches.Patch(color='red', label='Outlier (Score = 0)'),
        mpatches.Patch(color=plt.cm.Greens(0.4), label='Low Score (Low Trust)'),
        mpatches.Patch(color=plt.cm.Greens(0.8), label='High Score (High Trust)')
    ]
    ax.legend(handles=legend_handles, title="Client Status", fontsize=12, title_fontsize=14, loc='upper right')

    plt.tight_layout()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_path = os.path.join(output_dir, f"fedgraph_round_{round_idx}_optimized.png")
    plt.savefig(image_path, dpi=300)
    plt.close()
    logging.info(f"Optimized graph visualization saved to {image_path}")


    # now plot in the space of node features in 3d (because there are 3 features: degree, betweenness, closeness)
    # also add edges in this space
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Node color and size mapping
    node_colors = []
    node_sizes = []
    for i, client_id in enumerate(clients_this_round):
        score = scores[i]
        if labels[i] == -1:
            node_colors.append((1.0, 0.0, 0.0, 1.0))  # Outliers are red
            node_sizes.append(150)  # Fixed size for outliers
        else:
            normalized_score = score * 0.8 + 0.2 
            node_colors.append(plt.cm.Greens(normalized_score))
            node_sizes.append(250 + score * 400)  # Size increases with score

    print("len(node_features):", len(node_features))
    print("len(node_colors):", len(node_colors))
    print("len(node_sizes):", len(node_sizes))
    print("example node_feature[0]:", node_features[0])
    # breakpoint()

    # Scatter plot for node features
    ax.scatter(
        [nf[0] for nf in node_features],  # Degree
        [nf[1] for nf in node_features],  # Betweenness
        [nf[2] for nf in node_features],  # Closeness
        c=node_colors,
        s=node_sizes,
        alpha=0.9,
        edgecolors='black'
    )



    # Draw edges in 3D feature space
    for u, v, attrs in graph.edges(data=True):
        w = attrs.get('weight', 0.0)
        x_vals = [node_features[u][0], node_features[v][0]]
        y_vals = [node_features[u][1], node_features[v][1]]
        z_vals = [node_features[u][2], node_features[v][2]]
        ax.plot(x_vals, y_vals, z_vals, color='gray', alpha=0.6, linewidth=w * 7)

    # Add labels
    # for i, client_id in enumerate(clients_this_round):
    #     ax.text(
    #         node_features[i][0],
    #         node_features[i][1],
    #         node_features[i][2],
    #         f"C{client_id}\nScore: {scores[i]:.2f}",
    #         fontsize=10,
    #         fontfamily='sans-serif',
    #         fontweight='bold'
    #     )

    ax.set_title(f"FedGraph Node Features - Round {round_idx}", fontsize=20, fontweight='bold')
    ax.set_xlabel("Degree")
    ax.set_ylabel("Betweenness")
    ax.set_zlabel("Closeness")

    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_path = os.path.join(output_dir, f"fedgraph_round_{round_idx}_3d.png")
    plt.savefig(image_path, dpi=300)
    plt.close()
    logging.info(f"3D graph visualization saved to {image_path}")


def save_fedgraph_data(G, labels, scores, round_idx, clients_this_round, output_dir, node_features):
    """Saves graph and clustering data to a JSON file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    data = {
        'round_idx': round_idx,
        'nodes': [],
        'edges': [],
        'clustering': {
            'labels': labels.tolist(),
            'scores': scores.tolist(),
            'node_features': node_features
        }
    }
    
    # Add nodes with their attributes
    for i, client_id in enumerate(clients_this_round):
        data['nodes'].append({
            'id': int(i),
            'client_id': int(client_id),
            'label': int(labels[i]),
            'score': float(scores[i]),
            'node_features': node_features[i]
        })
    
    # Add edges with their weights
    for u, v, attrs in G.edges(data=True):
        data['edges'].append({
            'source': int(u),
            'target': int(v),
            'weight': float(attrs.get('weight', 0))
        })
        
    file_path = os.path.join(output_dir, f"fedgraph_data_round_{round_idx}.json")
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    logging.info(f"Graph data saved to {file_path}")

# --- Core Aggregation Function ---
def aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=None, output_dir=None):
    
    n_clients = len(clients_this_round)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Steps 1-3: Prepare data, build graph, and extract features (unchanged from your original code)
    client_updates = {}
    for client_idx, client in enumerate(clients_this_round):
        client_updates[client_idx] = {}
        for name, data in local_dict_list[client].items():
            update = data.add(-global_dict[name])
            client_updates[client_idx][name] = update
            
    layer_names = list(global_dict.keys())
    G = nx.Graph()
    G.add_nodes_from(range(n_clients))
    SIMILARITY_THRESHOLD = 0.8
    for name in layer_names:
        layer_updates = [client_updates[client_idx][name].flatten().cpu().numpy() for client_idx in range(n_clients)]
        cos_sim_matrix = smp.cosine_similarity(np.array(layer_updates)).astype(np.float64)
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                if cos_sim_matrix[i, j] > SIMILARITY_THRESHOLD:
                    G.add_edge(i, j, weight=cos_sim_matrix[i, j])

    logging.info(f"Graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges created.")

    if G.number_of_edges() > 0:
        node_degrees = dict(G.degree())
        betweenness = nx.betweenness_centrality(G, weight='weight')
        closeness = nx.closeness_centrality(G, distance='weight')
        max_degree = max(node_degrees.values()) if max(node_degrees.values()) > 0 else 1
        max_betweenness = max(betweenness.values()) if max(betweenness.values()) > 0 else 1
        max_closeness = max(closeness.values()) if max(closeness.values()) > 0 else 1
        node_features = []
        for i in range(n_clients):
            feature_vec = [
                node_degrees.get(i, 0) / max_degree,
                betweenness.get(i, 0) / max_betweenness,
                closeness.get(i, 0) / max_closeness
            ]
            node_features.append(feature_vec)
    else:
        node_features = [[0, 0, 0] for _ in range(n_clients)]
        logging.warning("No edges in the graph. Using default features.")

    # Step 4: HDBSCAN Clustering
    node_features_np = np.array(node_features).astype(np.float64)
    if node_features_np.shape[0] > 0 and node_features_np.shape[1] > 0:
        clusterer = hdbscan.HDBSCAN(min_samples=1, min_cluster_size=max(2, int(n_clients/2 + 1)), allow_single_cluster=True).fit(node_features_np)
        client_scores = clusterer.probabilities_
        cluster_labels = clusterer.labels_
        for client_idx, label in enumerate(cluster_labels):
            if label == -1:
                client_scores[client_idx] = 0.0
                logging.info(f"Client {clients_this_round[client_idx]} identified as a graph outlier. Score set to 0.")
    else:
        client_scores = np.ones(n_clients)
        cluster_labels = np.zeros(n_clients)
        logging.warning("No node features to cluster. Defaulting to score of 1.")

    # --- Save data for visualization ---
    if output_dir:
        save_fedgraph_data(G, cluster_labels, client_scores, round_idx, clients_this_round, output_dir, node_features)
        visualize_fedgraph(G, cluster_labels, client_scores, round_idx, output_dir, clients_this_round, node_features)
    
    
    
    total_score = np.sum(client_scores)
    final_weights = client_scores / total_score if total_score > 0 else np.zeros(n_clients)

    # Step 5: Weighted Aggregation (unchanged)
    weight_accumulator = {name: torch.zeros_like(global_dict[name]) for name in layer_names}
    for client_idx, client in enumerate(clients_this_round):
        weight = final_weights[client_idx]
        if weight > 0:
            for name, update in client_updates[client_idx].items():
                weighted_update = update * weight
                weight_accumulator[name].add_(weighted_update)
    for name in layer_names:
        global_dict[name].add_(weight_accumulator[name])
            
    return global_dict