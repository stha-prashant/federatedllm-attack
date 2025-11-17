import sklearn.metrics.pairwise as smp
import numpy as np
import os
import json

def aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=None, output_dir=None):
    n_clients = len(clients_this_round)

    method = 'cosine_clustering'
    # method = 'cosine_clustering'
    save_data  = {
        "round_idx": round_idx,
        "clients": clients_this_round,
        "whole_cosine_similarity": None,
        "layer_wise_cosine_similarity": {}
    }

    save_path = f"{output_dir}/cosine_similarity"
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"round_{round_idx}_cosine_similarity.json")
    client_updates = {}
    for client_idx, client in enumerate(clients_this_round):
        client_updates[client_idx] = {}
        for name, data in local_dict_list[client].items():
            update = data.add(-global_dict[name])
            client_updates[client_idx][name] = update
            
    layer_names = list(global_dict.keys())

    for name in layer_names:
        layer_updates = [client_updates[client_idx][name].flatten().cpu().numpy() for client_idx in range(n_clients)]
        cos_sim_matrix_layer = smp.cosine_similarity(np.array(layer_updates)).astype(np.float64)
        # print(f"---------Layer: {name}, shape: {cos_sim_matrix_layer.shape}---------")
        # print(cos_sim_matrix_layer)
        save_data['layer_wise_cosine_similarity'][name] = cos_sim_matrix_layer.tolist()
        # save to .npy
        
    client_update_vecs = []

    for client in clients_this_round:
        parts = []
        for name, local_tensor in local_dict_list[client].items():
            # (local - global) as 1D numpy
            upd = (local_tensor - global_dict[name]).detach()
            parts.append(upd.flatten().cpu().numpy())
        client_update_vecs.append(np.concatenate(parts, axis=0))

    # shape: (n_clients, D)
    all_updates = np.vstack(client_update_vecs)
    save_data["whole_cosine_similarity"] = smp.cosine_similarity(all_updates).tolist()

    with open(save_file, 'w') as f:
        json.dump(save_data, f)


    if method == 'fedavg':
        sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
        return global_dict
    elif method == 'cosine_clustering':
        from sklearn.cluster import SpectralClustering
        S = np.array(save_data['whole_cosine_similarity'], dtype=np.float64)
        A = (S + 1.0) / 2.0
        np.fill_diagonal(A, 1.0)

        clu = SpectralClustering(
            n_clusters=2,
            affinity='precomputed',        # we pass A directly
            assign_labels='kmeans',        # sklearn’s default discretization
            random_state=0
        ).fit(A)

        labels = clu.labels_
        # pick the larger cluster
        vals, counts = np.unique(labels, return_counts=True)
        keep_label = vals[np.argmax(counts)]
        kept_idx = np.where(labels == keep_label)[0]
        kept_client_ids = [clients_this_round[i] for i in kept_idx]

        sample_this_round = sum([sample_num_list[client] for client in kept_client_ids])
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in kept_client_ids])
        return global_dict