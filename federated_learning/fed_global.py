import random
import torch
#== attack and defense ==
import math
import copy
from functools import reduce
import numpy as np
import torch.nn.functional as F 
import os
import json
#== attack and defense ==

def get_clients_this_round(fed_args, round):
    if (fed_args.fed_alg).startswith('local'):
        clients_this_round = [int((fed_args.fed_alg)[-1])]
    else:
        if fed_args.num_clients < fed_args.sample_clients:
            clients_this_round = list(range(fed_args.num_clients))
        else:
            random.seed(round)
            clients_this_round = sorted(random.sample(range(fed_args.num_clients), fed_args.sample_clients))
    return clients_this_round

def global_aggregate(fed_args, global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, proxy_dict=None, opt_proxy_dict=None, auxiliary_info=None, base_model_path=None, project_matrix=None, script_args=None, asr_rates=None, project_matrix_edit=None, safety_gradient_by_param=None):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    global_auxiliary = None

    if fed_args.fed_alg == 'scaffold':
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
        global_auxiliary, auxiliary_delta_dict = auxiliary_info
        for key in global_auxiliary.keys():
            delta_auxiliary = sum([auxiliary_delta_dict[client][key] for client in clients_this_round]) 
            global_auxiliary[key] += delta_auxiliary / fed_args.num_clients
    
    elif fed_args.fed_alg == 'fedavgm':
        # Momentum-based FedAvg
        for key in global_dict.keys():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round for client in clients_this_round])
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            global_dict[key] = global_dict[key] + proxy_dict[key]

    elif fed_args.fed_alg == 'fedadagrad':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            # In paper 'adaptive federated optimization', momentum is not used
            proxy_dict[key] = delta_w
            opt_proxy_dict[key] = param + torch.square(proxy_dict[key])
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'fedyogi':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            delta_square = torch.square(proxy_dict[key])
            opt_proxy_dict[key] = param - (1-fed_args.fedopt_beta2)*delta_square*torch.sign(param - delta_square)
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'fedadam':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            opt_proxy_dict[key] = fed_args.fedopt_beta2*param + (1-fed_args.fedopt_beta2)*torch.square(proxy_dict[key])
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    
    # ============== Defense baselines ==================
    elif fed_args.fed_alg == 'median':
        key_list = {}
        for net_id, client in enumerate(clients_this_round):
            net_para = local_dict_list[client]
            if net_id == 0:
                for key in net_para:
                    key_list[key] = [net_para[key].unsqueeze(0)]
            else:
                for key in net_para:
                    key_list[key].append(net_para[key].unsqueeze(0))
        for key in net_para:
            key_value_cat = torch.cat(key_list[key])
            key_value_median, _ = torch.median(key_value_cat, dim=0)
            global_dict[key] = key_value_median
    
    elif fed_args.fed_alg == 'krum' :
        expected_n_attacker = 0
        for malicious_num_client in fed_args.malicious_num_clients:
            expected_n_attacker += malicious_num_client
        
        expected_n_attacker = round(fed_args.sample_clients*expected_n_attacker/fed_args.num_clients)
     
        model_weight_list = []
        for net_id, client in enumerate(clients_this_round):
            net_para = local_dict_list[client]
            model_weight = get_weight(net_para).unsqueeze(0)
            model_weight_list.append(model_weight)
        model_weight_cat = torch.cat(model_weight_list, dim=0)
        model_weight_krum, aggregate_idx, nbhdist, nbh = get_krum(model_weight_cat, expected_n_attacker)
        model_weight_krum = model_weight_krum.reshape(-1)
        # make sure aggregate_idx is on CPU, because clients_this_round is on CPU
        aggregate_idx = aggregate_idx.cpu() if aggregate_idx.is_cuda else aggregate_idx

        aggregate_idx_list = torch.tensor(clients_this_round)[aggregate_idx].tolist()

        # save selected clients to file for analysis
        path = os.path.join(script_args.output_dir, 'krum')
        save_data = {
            'round_idx': round_idx,
            'selected_clients': aggregate_idx_list,
            'nbhdist': nbhdist.cpu().tolist(),
            'nbh': nbh.cpu().tolist()
        }
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f'round_{round_idx}.json'), 'w') as f:
            json.dump(save_data, f)
        aggregate_idx_list.sort()

        current_idx = 0
        for key in net_para:
            length = len(net_para[key].reshape(-1))
            global_dict[key] = model_weight_krum[current_idx : current_idx + length].reshape(net_para[key].shape)
            current_idx += length
        print(f"===> Krum selected client: {aggregate_idx_list} in round {round_idx}")

    elif fed_args.fed_alg == 'krumoriginal':
        expected_n_attacker = 0
        for malicious_num_client in fed_args.malicious_num_clients:
            expected_n_attacker += malicious_num_client
        
        expected_n_attacker = round(fed_args.sample_clients*expected_n_attacker/fed_args.num_clients)

        model_weight_list = []
        for net_id, client in enumerate(clients_this_round):
            net_para = local_dict_list[client]
            model_weight = get_weight(net_para).unsqueeze(0)
            model_weight_list.append(model_weight)
        model_weight_cat = torch.cat(model_weight_list, dim=0)
        model_weight_krum, aggregate_idx, neighbours_list = get_krum_original(model_weight_cat, expected_n_attacker)
        model_weight_krum = model_weight_krum.reshape(-1)
        # make sure aggregate_idx is on CPU, because clients_this_round is on CPU
        aggregate_idx = [aggregate_idx]

        aggregate_idx_list = torch.tensor(clients_this_round)[aggregate_idx].tolist()

        # save selected clients to file for analysis
        path = os.path.join(script_args.output_dir, 'krumoriginal')
        save_data = {
            'round_idx': round_idx,
            'selected_clients': aggregate_idx_list,
            'closest_neighbours': neighbours_list
        }
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f'round_{round_idx}.json'), 'w') as f:
            json.dump(save_data, f)
        aggregate_idx_list.sort()

        current_idx = 0
        for key in net_para:
            length = len(net_para[key].reshape(-1))
            global_dict[key] = model_weight_krum[current_idx : current_idx + length].reshape(net_para[key].shape)
            current_idx += length
        print(f"===> Krum original selected client: {aggregate_idx_list[0]} in round {round_idx}")
    elif fed_args.fed_alg == 'multikrum':
        expected_n_attacker = 0
        for malicious_num_client in fed_args.malicious_num_clients:
            expected_n_attacker += malicious_num_client
        
        expected_n_attacker = round(fed_args.sample_clients*expected_n_attacker/fed_args.num_clients)

        model_weight_list = []
        for net_id, client in enumerate(clients_this_round):
            net_para = local_dict_list[client]
            model_weight = get_weight(net_para).unsqueeze(0)
            model_weight_list.append(model_weight)
        model_weight_cat = torch.cat(model_weight_list, dim=0)
        model_weight_krum, aggregate_idx, scores = get_multikrum_original(model_weight_cat, expected_n_attacker)
        model_weight_krum = model_weight_krum.reshape(-1)
        # make sure aggregate_idx is on CPU, because clients_this_round is on CPU
        aggregate_idx = aggregate_idx

        aggregate_idx_list = torch.tensor(clients_this_round)[aggregate_idx].tolist()

        # save selected clients to file for analysis
        path = os.path.join(script_args.output_dir, 'multikrum')
        save_data = {
            'round_idx': round_idx,
            'selected_clients': aggregate_idx_list,
            'scores': scores.cpu().tolist()
        }
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f'round_{round_idx}.json'), 'w') as f:
            json.dump(save_data, f)
        aggregate_idx_list.sort()

        current_idx = 0
        for key in net_para:
            length = len(net_para[key].reshape(-1))
            global_dict[key] = model_weight_krum[current_idx : current_idx + length].reshape(net_para[key].shape)
            current_idx += length
    
    elif fed_args.fed_alg == 'trimmedmean':
        net_para_list = []
        for net_id, client in enumerate(clients_this_round):
            net_para_list.append(local_dict_list[client])
            
        trimmed_num = 1
        
        # Trimmed mean
        for key in global_dict:
            net_para_stack = torch.stack([net_row[key] for net_row in net_para_list])
            net_shape = net_para_stack.shape[1:]
            net_para_stack = net_para_stack.reshape(len(net_para_list), -1)
            net_para_sorted = net_para_stack.sort(dim=0).values
            result = net_para_sorted[trimmed_num:-trimmed_num, :]
            result_type = result.dtype
            result = result.float().mean(dim=0).type(result_type)
            result = result.reshape(net_shape)
            global_dict[key] = result
    
    elif fed_args.fed_alg == 'foolsgold':
        local_dict_list_this_round = [local_dict_list[i] for i in clients_this_round]
        model_weight_list = []
        for net_id, net_para in enumerate(local_dict_list_this_round):
            model_weight = get_weight(net_para).unsqueeze(0)
            model_weight_list.append(model_weight)
        model_weight_cat = torch.cat(model_weight_list, dim=0)

        update_mean, update_std, update_cat, global_weight = get_update_static(local_dict_list_this_round, global_dict)
        model_weight_foolsgold, wv, cs = get_foolsgold(update_cat, global_weight)

        current_idx = 0 
        # save client weights for analysis
        path = os.path.join(script_args.output_dir, 'foolsgold')
        os.makedirs(path, exist_ok=True)
        save_data = {
            'round_idx': round_idx,
            'client_weights': wv.cpu().tolist(),
            'cs': cs.cpu().tolist()
        }
        with open(os.path.join(path, f'round_{round_idx}.json'), 'w') as f:
            json.dump(save_data, f)

        for key in net_para:
            length = len(net_para[key].reshape(-1))
            global_dict[key] = model_weight_foolsgold[current_idx : current_idx + length].reshape(net_para[key].shape)
            current_idx += length   
        
    elif fed_args.fed_alg == 'foolsgoldbenign':
        # just use the weights used by the reference
        reference_path = os.path.join(fed_args.foolsgoldbenign_reference, 'foolsgold', f'round_{round_idx}.json')
        with open(reference_path, 'r') as f:
            reference_data = json.load(f)
        # save the same weights in this checkpoint folder
        path = os.path.join(script_args.output_dir, 'foolsgoldbenign')
        os.makedirs(path, exist_ok=True)

        
        reference_data = reference_data['client_weights']
        # treat last 3 as malicious clients and set weights to 0, otherwise use the reference weights
        reference_weights = torch.tensor(reference_data)
        reference_weights[-3:] = 0.0

        local_dict_list_this_round = [local_dict_list[i] for i in clients_this_round]
        model_weight_list = []
        for net_id, net_para in enumerate(local_dict_list_this_round):
            model_weight = get_weight(net_para).unsqueeze(0)
            model_weight_list.append(model_weight)
        model_weight_cat = torch.cat(model_weight_list, dim=0)

        update_mean, update_std, update_cat, global_weight = get_update_static(local_dict_list_this_round, global_dict)
        model_weight_foolsgold, wv = get_foolsgoldbenign(update_cat, global_weight, wv=reference_weights)

        current_idx = 0

        for key in net_para:
            length = len(net_para[key].reshape(-1))
            global_dict[key] = model_weight_foolsgold[current_idx : current_idx + length].reshape(net_para[key].shape)
            current_idx += length

        save_data = {
            'round_idx': round_idx,
            'client_weights': reference_weights.cpu().tolist()
        }
        with open(os.path.join(path, f'round_{round_idx}.json'), 'w') as f:
            json.dump(save_data, f)
        
    
    elif fed_args.fed_alg == 'residual':
        local_dict_list_this_round = [local_dict_list[i] for i in clients_this_round]
        model_weight_list = []
        global_dict, reweight = IRLS_aggregation_split_restricted(local_dict_list_this_round, 2.0, 0.05)

    # elif fed_args.fed_alg == 'evaladvbench':
    #     from .evaladvbench import aggr
    #     global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/evaladvbench/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]')
    elif fed_args.fed_alg == 'oracle':
        clients_this_round = [0, 1, 2, 3, 4]
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])

    elif fed_args.fed_alg == 'dnc':
        local_dict_list_this_round = [local_dict_list[i] for i in clients_this_round]
        expected_n_attacker = 0
        for malicious_num_client in fed_args.malicious_num_clients:
            expected_n_attacker += malicious_num_client
            
        expected_n_attacker = round(fed_args.sample_clients*expected_n_attacker/fed_args.num_clients)
            
        current_idx = 0
        update_mean, update_std, update_cat, global_weight = get_update_static(local_dict_list_this_round, global_dict)
        i_final,wv, outlier_score = do_dnc(update_cat,m=expected_n_attacker)
        print("===> DnC Aggregation: ", wv)

        path = os.path.join(script_args.output_dir, 'dnc')
        save_data = {
            'round_idx': round_idx,
            'selected_clients': i_final.tolist(),
            'outlier_score': outlier_score.tolist()
        }
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f'round_{round_idx}.json'), 'w') as f:
            json.dump(save_data, f)

        
        model_weight_foolsgold = foolsgold_wv_update(wv, update_cat, global_weight)
        
        for net_id, net_para in enumerate(local_dict_list_this_round):
            break
        
        for key in net_para:
            length = len(net_para[key].reshape(-1))
            global_dict[key] = model_weight_foolsgold[current_idx : current_idx + length].reshape(net_para[key].shape)
            current_idx += length
        
    elif fed_args.fed_alg == 'fedgraph':
        from .safelayer_vis import aggr
        output_dir = os.path.join(script_args.output_dir, 'fedgraph')
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict,output_dir=output_dir)

        # global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict,output_dir=f'./output/fedgraph/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]')
    
    elif fed_args.fed_alg == 'lasa':
        local_dict_list_this_round = [local_dict_list[i] for i in clients_this_round]

        def get_trainable_float_keys(net_dict):
            keys = []
            for key, param in net_dict.items():
                if 'num_batches_tracked' in key:
                    continue
                if not torch.is_floating_point(param):
                    continue
                keys.append(key)
            return keys

        def dict_to_vec(net_dict, keys):
            return torch.cat([net_dict[key].reshape(-1) for key in keys])

        @torch.no_grad()
        def vec_to_dict_(vec, net_dict, keys):
            pointer = 0
            for key in keys:
                param = net_dict[key]
                num_param = param.numel()
                param.copy_(vec[pointer:pointer + num_param].view_as(param))
                pointer += num_param

        @torch.no_grad()
        def generate_init_mask(model_dict, keys):
            mask = {}
            for key in keys:
                param = model_dict[key]
                if len(param.size()) == 4 or len(param.size()) == 2:
                    mask[key] = torch.ones_like(param, dtype=torch.float32, requires_grad=False)
            return mask

        @torch.no_grad()
        def update_mask_by_topk_(model, mask, keys, sparsity):
            if sparsity == 0.0:
                for key in keys:
                    if key in mask:
                        mask[key] = torch.ones_like(mask[key]).float()
                return mask

            weight_abs = []
            for key in keys:
                if key not in mask:
                    continue
                weight_abs.append(model[key].abs())

            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(len(all_scores) * (1 - sparsity))

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for key in keys:
                if key not in mask:
                    continue
                mask[key] = (model[key].abs() > acceptable_score).float()

            return mask

        @torch.no_grad()
        def apply_mask_(model, mask, keys):
            for key in keys:
                if key in mask:
                    model[key].mul_(mask[key])
            return model

        @torch.no_grad()
        def add_update_(global_model, update, keys):
            for key in keys:
                global_model[key].add_(update[key])

        @torch.no_grad()
        def has_nan(update, keys):
            for key in keys:
                if torch.isnan(update[key]).any():
                    return True
            return False

        @torch.no_grad()
        def robust_zscore_mask(x, thr):
            med = x.median()
            std = x.std(unbiased=False)
            z = (x - med).abs() / std
            return z < thr

        @torch.no_grad()
        def layer_sign_score(t, sparsity):
            s = torch.sign(t)
            nz = s != 0
            denom = nz.sum().clamp_min(1)
            balance = s.sum() / denom
            return 0.5 * (1 + balance * (1 - sparsity))

        def lasa(local_updates, global_model, args):
            all_keys = get_trainable_float_keys(global_model)
            mask_keys = [key for key in all_keys if len(global_model[key].size()) in (2, 4)]

            local_updates_ = []
            valid_indices = []
            for i in range(len(local_updates)):
                if has_nan(local_updates[i], all_keys):
                    continue
                local_updates_.append(local_updates[i])
                valid_indices.append(i)

            local_updates = local_updates_
            if len(local_updates) == 0:
                return global_model, {
                    'valid_indices': [],
                    'benign_layer_counts': [],
                    'n_mask_keys': len(mask_keys),
                }

            flat_local_updates = [dict_to_vec(u, all_keys) for u in local_updates]

            flat_all_grads = torch.stack(flat_local_updates, dim=0)
            grad_norm = torch.norm(flat_all_grads, dim=1).reshape((-1, 1))
            norm_clip = grad_norm.median(dim=0)[0].item()
            grad_norm_clipped = torch.clamp(grad_norm, 0, norm_clip, out=None)
            grads_clip = (flat_all_grads / (grad_norm)) * grad_norm_clipped

            for i in range(len(local_updates)):
                vec_to_dict_(grads_clip[i], local_updates[i], all_keys)

            if len(mask_keys) > 0:
                for i in range(len(local_updates)):
                    global_mask = generate_init_mask(local_updates[i], mask_keys)
                    global_mask = update_mask_by_topk_(local_updates[i], global_mask, mask_keys, args.sparsity)
                    local_updates[i] = apply_mask_(local_updates[i], global_mask, mask_keys)

            key_mean_weight = {}
            n = len(local_updates)
            benign_layer_counts = torch.zeros(n, dtype=torch.int, device=local_updates[0][mask_keys[0]].device)
            for key in mask_keys:
                grads = torch.stack([local_updates[i][key] for i in range(n)], dim=0)
                norms = grads.float().reshape(n, -1).norm(dim=1)
                benign1 = robust_zscore_mask(norms, args.lambda_n)

                scores = torch.stack([layer_sign_score(local_updates[i][key], args.sparsity) for i in range(n)], dim=0)
                benign2 = robust_zscore_mask(scores, args.lambda_s)

                benign = benign1 & benign2
                benign_layer_counts += benign.int()
                idx = benign.nonzero(as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    idx = torch.arange(n, device=grads.device)

                key_mean_weight[key] = grads[idx].mean(dim=0)

            if len(mask_keys) > 0:
                add_update_(global_model, key_mean_weight, mask_keys)

            return global_model, {
                'valid_indices': valid_indices,
                'benign_layer_counts': benign_layer_counts.cpu().tolist(),
                'n_mask_keys': len(mask_keys),
            }

        update_params = []
        for net_para in local_dict_list_this_round:
            update = {}
            for key in net_para.keys():
                update[key] = (net_para[key] - global_dict[key]).clone()
            update_params.append(update)

        sparsity = getattr(fed_args, 'sparsity', 0.3)
        lambda_n = getattr(fed_args, 'lambda_n', 1.0)
        lambda_s = getattr(fed_args, 'lambda_s', 1.0)
        num_selected_users = len(update_params)

        class _LasaArgs:
            pass

        lasa_args = _LasaArgs()
        lasa_args.sparsity = sparsity
        lasa_args.lambda_n = lambda_n
        lasa_args.lambda_s = lambda_s
        lasa_args.num_selected_users = num_selected_users

        global_dict, lasa_stats = lasa(update_params, global_dict, lasa_args)

        n_mask_keys = lasa_stats['n_mask_keys']
        valid_indices = lasa_stats['valid_indices']
        benign_layer_counts = lasa_stats['benign_layer_counts']
        if n_mask_keys > 0:
            selected_clients = [
                clients_this_round[valid_indices[i]]
                for i, cnt in enumerate(benign_layer_counts)
                if cnt == n_mask_keys
            ]
            layers_benign_fraction = {
                clients_this_round[valid_indices[i]]: cnt / n_mask_keys
                for i, cnt in enumerate(benign_layer_counts)
            }
        else:
            selected_clients = list(clients_this_round)
            layers_benign_fraction = {client: 1.0 for client in clients_this_round}

        path = os.path.join(script_args.output_dir, 'lasa')
        save_data = {
            'round_idx': round_idx,
            'clients': clients_this_round,
            'selected_clients': selected_clients,
            'layers_benign_fraction': layers_benign_fraction,
        }
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f'round_{round_idx}.json'), 'w') as f:
            json.dump(save_data, f)

    elif fed_args.fed_alg == 'flame':
        local_dict_list_this_round = [local_dict_list[i] for i in clients_this_round]
        def vector_lora(net_dict):
            vec = []
            for key, param in net_dict.items():
                if "lora_" not in key:
                    continue
                vec.append(param.view(-1))
            return torch.cat(vec)

        update_params = []
        for net_para in local_dict_list_this_round:
            update = {}
            for key in net_para.keys():
                update[key] = net_para[key] - global_dict[key]
            update_params.append(update)

        cos_list = []
        local_model_vector = [vector_lora(param) for param in local_dict_list_this_round]
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).to(local_model_vector[0].device)
        for i in range(len(local_model_vector)):
            cos_i = []
            for j in range(len(local_model_vector)):
                cos_ij = 1 - cos(local_model_vector[i], local_model_vector[j])
                cos_i.append(cos_ij.item())
            cos_list.append(cos_i)

        n_clients = len(local_dict_list_this_round)
        min_cluster_size = n_clients // 2 + 1
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            allow_single_cluster=True
        ).fit(cos_list)
        labels = clusterer.labels_

        benign_client = []
        max_num_in_cluster = 0
        max_cluster_index = 0
        if labels.max() < 0:
            benign_client = list(range(n_clients))
        else:
            for index_cluster in range(labels.max() + 1):
                if len(labels[labels == index_cluster]) > max_num_in_cluster:
                    max_cluster_index = index_cluster
                    max_num_in_cluster = len(labels[labels == index_cluster])
            for i in range(len(labels)):
                if labels[i] == max_cluster_index:
                    benign_client.append(i)

        norm_list = np.array([])
        for i in range(len(local_model_vector)):
            norm_list = np.append(norm_list, torch.norm(vector_lora(update_params[i]), p=2).item())

        clip_value = np.median(norm_list)
        for i in range(len(benign_client)):
            gamma = clip_value / norm_list[benign_client[i]]
            if gamma < 1:
                for key in update_params[benign_client[i]]:
                    if key.split('.')[-1] == 'num_batches_tracked':
                        continue
                    update_params[benign_client[i]][key] *= gamma

        total_num = len(benign_client)
        sum_parameters = None
        for idx in benign_client:
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in update_params[idx].items():
                    sum_parameters[key] = var.clone()
            else:
                for key in sum_parameters:
                    sum_parameters[key] = sum_parameters[key] + update_params[idx][key]
        for key in global_dict:
            if key.split('.')[-1] == 'num_batches_tracked':
                global_dict[key] = update_params[benign_client[0]][key]
                continue
            global_dict[key] += (sum_parameters[key] / total_num)

        noise = 0.001
        for key, var in global_dict.items():
            if key.split('.')[-1] == 'num_batches_tracked':
                continue
            temp = copy.deepcopy(var)
            temp = temp.normal_(mean=0, std=noise * clip_value)
            var += temp

        path = os.path.join(script_args.output_dir, 'flame')
        save_data = {
            'round_idx': round_idx,
            'clients': clients_this_round,
            'selected_clients': [clients_this_round[i] for i in benign_client],
            'cluster_labels': labels.tolist(),
            'norm_list': norm_list.tolist(),
        }
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f'round_{round_idx}.json'), 'w') as f:
            json.dump(save_data, f)

    elif fed_args.fed_alg == 'cosine_clustering':
        from .cosine_clustering import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/cosine_clustering/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]')
    elif fed_args.fed_alg == 'safefedllm':
        # Soft weights already encoded in sample_num_list (effective_samples from Safe-FedLLM).
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
    elif fed_args.fed_alg == 'eval_filter':
        from .eval_filter import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/evalfilter/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', script_args=script_args, asr_rates=asr_rates)
    elif fed_args.fed_alg == 'safe_lora':
        from .safelora import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelora/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, script_args=script_args)
    elif fed_args.fed_alg == 'safelorav2data':
        from .safelorav2 import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelorav2/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, script_args=script_args)
    elif fed_args.fed_alg == 'safelorav2dataadaptive':
        from .safelorav2dataadaptive import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelorav2dataadaptive/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, script_args=script_args)
    elif fed_args.fed_alg == 'safelorav2dataadaptiveold':
        from .safelorav2dataadaptiveold import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelorav2dataadaptiveold/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, script_args=script_args)
    elif fed_args.fed_alg == 'safelorav2datacontrast':
        # Full displacement (like safelorav2data) + contrastllmlat reference matrix.
        from .safelorav2datacontrast import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelorav2datacontrast/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, script_args=script_args)
    elif fed_args.fed_alg == 'safelorav2dataround':
        from .safelorav2dataround import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelorav2dataround/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, script_args=script_args)
    elif fed_args.fed_alg == 'safelorav2dataroundnoatt':
        from .safelorav2dataroundnoatt import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelorav2dataroundnoatt/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, script_args=script_args)
    elif fed_args.fed_alg == 'safelorav2dataroundbenignlatch':
        from .safelorav2dataroundbenignlatch import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelorav2dataroundbenignlatch/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, script_args=script_args)
    elif fed_args.fed_alg == 'safelorav2dataroundscoreonly':
        from .safelorav2dataroundscoreonly import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, project_matrix=project_matrix, script_args=script_args)
    elif fed_args.fed_alg == 'safelorav2dataroundmultiple':
        from .safelorav2dataroundmultiple import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelorav2dataroundmultiple/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, script_args=script_args)
    elif fed_args.fed_alg == 'safelorav2datafullcos':
        from .safelorav2datafullcos import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelorav2datafullcos/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, script_args=script_args)
    elif fed_args.fed_alg == 'safelorav2datadiff':
        from .safelorav2datadiff import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelorav2datadiff/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, project_matrix_edit=project_matrix_edit, script_args=script_args)
    elif fed_args.fed_alg == 'safelorav2datadiffround':
        from .safelorav2datadiffround import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelorav2datadiffround/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, project_matrix_edit=project_matrix_edit, script_args=script_args)
    elif fed_args.fed_alg == 'safelorav2datanomal':
        from .safelorav2nomal import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelorav2datanomal/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, script_args=script_args)
    elif fed_args.fed_alg == 'safelorav2datanormal2':
        from .safelorav2datanormal2 import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelorav2datanormal2/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, script_args=script_args)
    elif fed_args.fed_alg == 'smoketestanchor':
        from .smoketestanchor import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/smoketestanchor/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, script_args=script_args)
    elif fed_args.fed_alg == 'safeloradotdata':
        from .safeloradot import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safeloradot/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, script_args=script_args)
    elif fed_args.fed_alg == 'safelorav2referenceweighted':
        from .safelorav2_referenceweighted import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelorav2referenceweighted/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, script_args=script_args)
    elif fed_args.fed_alg == 'safelorav2randomweighted':
        from .safelorav2_referenceweighted import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelorav2referenceweighted/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, script_args=script_args, shuffle_benign_weights=True)

    elif fed_args.fed_alg == 'safelorav2warmup':
        from .safelorav2warmup import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelorav2/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, script_args=script_args)
    elif fed_args.fed_alg == 'safe_lora_mixture':
        from .safelora_mixture import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelora_mixture/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, script_args=script_args)
    elif fed_args.fed_alg == 'safe_lora_mixture_layerwise':
        from .safelora_mixture_layerwise import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelora_mixture_layerwise/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, script_args=script_args)
    elif fed_args.fed_alg == 'safe_lora_mixture_analytical':
        from .safelora_mixture_analytical import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelora_mixture_analytical/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, script_args=script_args)
    elif fed_args.fed_alg == 'safe_lora_mixture_analytical_oracle':
        from .safelora_mixture_analytical_oracle import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelora_mixture_analytical_oracle/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, script_args=script_args)
    
    elif fed_args.fed_alg == 'safe_lora_mixture_analytical_different':
        from .safelora_mixture_analytical_different import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelora_mixture_analytical_different/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, project_matrix_edit=project_matrix_edit, script_args=script_args)
    
    elif fed_args.fed_alg == 'safe_lora_mixture_analytical_different_old':
        from .safelora_mixture_analytical_different import aggr
        global_dict = aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=proxy_dict, output_dir=f'./output/safelora_mixture_analytical_different_old/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]', project_matrix=project_matrix, project_matrix_edit=project_matrix_edit, script_args=script_args)
    elif fed_args.fed_alg == 'safe_lora_mixture_safety_subspace':
        from .safelora_mixture_safety_subspace import aggr
        global_dict = aggr(
            global_dict,
            local_dict_list,
            sample_num_list,
            clients_this_round,
            round_idx,
            fed_args,
            proxy_dict=proxy_dict,
            output_dir=f'./output/safelora_mixture_safety_subspace/{base_model_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}]',
            project_matrix=project_matrix,
            script_args=script_args,
            safety_gradient_by_param=safety_gradient_by_param,
        )
    
    elif fed_args.fed_alg == 'debug_thrown':

        clients_this_round = [0, 1, 2, 3, 4,  5, 6]
        clients_this_round = [client for client in clients_this_round if client != script_args.throw_n]
        sample_this_round = sum([sample_num_list[client] for client in clients_this_round])

        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
            # global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round if client in [0, 1, 2, 3, 4]])
    else:   # Normal dataset-size-based aggregation 
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
    
    return global_dict, global_auxiliary

def get_weight(model_weight):
    weight_tensor_result = []
    for k, v in model_weight.items():
        weight_tensor_result.append(v.reshape(-1).float())
    weights = torch.cat(weight_tensor_result)
    return weights

def get_krum(inputs, attacker_num=1):
    inputs = inputs.unsqueeze(0).permute(0, 2, 1)
    n = inputs.shape[-1]
    k = n - attacker_num - 1

    x = inputs.permute(0, 2, 1)

    cdist = torch.cdist(x, x, p=2)
    # find the k+1 nbh of each point
    nbhDist, nbh = torch.topk(cdist, k + 1, largest=False)
    # the point closest to its nbh
    i_star = torch.argmin(nbhDist.sum(2))
    mkrum = inputs[:, :, nbh[:, i_star, :].view(-1)].mean(2, keepdims=True)
    return mkrum, nbh[:, i_star, :].view(-1), nbhDist, nbh

def get_krum_original(inputs, attacker_num=1):
    n = inputs.shape[0]
    k = n - attacker_num - 2

    cdist = torch.cdist(inputs, inputs, p=2)
    # find the k+1 nbh of each point
    nbhDist, nbh = torch.topk(cdist, k + 1, largest=False)
    # the point closest to its nbh
    i_star = torch.argmin(nbhDist.sum(1))
    krum = inputs[i_star, :]
    return krum, i_star.cpu().item(), nbh[i_star, :].view(-1).cpu().tolist()

def get_multikrum_original(inputs, attacker_num=1):
    n = inputs.shape[0]
    k = n - attacker_num - 2
    m = k  # common choice

    cdist = torch.cdist(inputs, inputs, p=2)
    cdist.fill_diagonal_(float("inf"))          # exclude self
    nbhDist, _ = torch.topk(cdist, k, largest=False)
    scores = nbhDist.sum(1)                      # krum score for each client

    selected = torch.topk(scores, m, largest=False).indices
    multikrum = inputs[selected].mean(0, keepdims=True)

    return multikrum, selected.cpu().tolist(), scores.cpu()

def get_update_static(local_dict_list, global_dict):
    model_weight_list = []
    net_id_list = []

    glboal_net_para = global_dict
    global_weight = get_weight(glboal_net_para).unsqueeze(0)

    for net_id, net_para in enumerate(local_dict_list):
        net_id_list.append(net_id)
        model_weight = get_weight(net_para).unsqueeze(0)
        model_update = model_weight - global_weight
        model_weight_list.append(model_update)
    model_weight_cat = torch.cat(model_weight_list, dim=0)
    model_std, model_mean = torch.std_mean(model_weight_cat, unbiased=False, dim=0)

    return model_mean, model_std, model_weight_cat, global_weight

def get_foolsgold(grads, global_weight):
    n_clients = grads.shape[0]
    grads_norm = F.normalize(grads, dim=1)
    device = grads.device # make sure aggregate_idx is on CPU,because clients_this_round is on CPU
    cs = torch.mm(grads_norm, grads_norm.T)
    cs = cs - torch.eye(n_clients, device=device)
    maxcs, _ = torch.max(cs, axis=1)

    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    maxcs_2, _ = torch.max(cs, axis=1)
    wv = 1 - maxcs_2

    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / torch.max(wv)
    wv[(wv == 1)] = 0.99

    # Logit function
    eps = 1e-8  # Add a small constant to prevent division by zero 
    wv = torch.log(wv / (1 - wv + eps)) + 0.5
    wv[(torch.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    model_weight_list = []
    for i in range(0, n_clients):
        if wv[i] != 0:
            current_weight = global_weight + wv[i] * grads[i]
            model_weight_list.append(current_weight.to(device))
    fools_gold_weight = torch.cat(model_weight_list).mean(0, keepdims=True)

    return fools_gold_weight.view(-1), wv, cs


def get_foolsgoldbenign(grads, global_weight, wv):
    n_clients = grads.shape[0]
    device = grads.device # make sure aggregate_idx is on CPU,because clients_this_round is on CPU

    wv = wv.to(device)
    model_weight_list = []
    for i in range(0, n_clients):
        if wv[i] != 0:
            current_weight = global_weight + wv[i] * grads[i]
            model_weight_list.append(current_weight.to(device))
    fools_gold_weight = torch.cat(model_weight_list).mean(0, keepdims=True)

    return fools_gold_weight.view(-1), wv

def IRLS_aggregation_split_restricted(local_dict_list, LAMBDA=2, thresh=0.1):
    SHARD_SIZE = 2000

    w = []
    for net_id, net_para in enumerate(local_dict_list):
        w.append(net_para)

    w_med = copy.deepcopy(w[0])

    device = w[0][list(w[0].keys())[0]].device
    reweight_sum = torch.zeros(len(w)).to(device)

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        transposed_y_list = torch.t(y_list)
        y_result = torch.zeros_like(transposed_y_list)

        if total_num < SHARD_SIZE:
            reweight, restricted_y = reweight_algorithm_restricted(transposed_y_list, LAMBDA, thresh)
            reweight_sum += reweight.sum(dim=0)
            y_result = restricted_y
        else:
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y = transposed_y_list[i * SHARD_SIZE : (i + 1) * SHARD_SIZE, ...]
                reweight, restricted_y = reweight_algorithm_restricted(y, LAMBDA, thresh)
                reweight_sum += reweight.sum(dim=0)
                y_result[i * SHARD_SIZE : (i + 1) * SHARD_SIZE, ...] = restricted_y

        # put restricted y back to w
        y_result = torch.t(y_result)
        for i in range(len(w)):
            w[i][k] = y_result[i].reshape(w[i][k].shape).to(device)

    reweight_sum = reweight_sum / reweight_sum.max()
    reweight_sum = reweight_sum * reweight_sum
    w_med, reweight = weighted_average(w, reweight_sum)

    return w_med, reweight

def reweight_algorithm_restricted(y, LAMBDA, thresh):
    num_models = y.shape[1]
    total_num = y.shape[0]
    slopes, intercepts = repeated_median(y)
    X_pure = y.sort()[1].sort()[1].type(torch.float)

    # calculate H matrix
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)
    X_X = torch.matmul(X.transpose(1, 2), X)
    X_X = torch.matmul(X, torch.inverse(X_X))
    H = torch.matmul(X_X, X.transpose(1, 2))
    diag = torch.eye(num_models).repeat(total_num, 1, 1).to(y.device)
    processed_H = (torch.sqrt(1 - H) * diag).sort()[0][..., -1]
    K = torch.FloatTensor([LAMBDA * np.sqrt(2.0 / num_models)]).to(y.device)

    beta = torch.cat(
        (
            intercepts.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
            slopes.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
        ),
        dim=-1,
    )
    line_y = (beta * X).sum(dim=-1)
    residual = y - line_y
    M = median_opt(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M + 1e-7
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)
    reweight = processed_H / e * torch.max(-K, torch.min(K, e / processed_H))
    reweight[reweight != reweight] = 1
    reweight_std = reweight.std(dim=1)  # its standard deviation
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = reweight * reshaped_std  # reweight confidence by its standard deviation

    restricted_y = y * (reweight >= thresh) + line_y * (reweight < thresh)
    return reweight_regulized, restricted_y

def median_opt(input):
    shape = input.shape
    input = input.sort()[0]
    if shape[-1] % 2 != 0:
        output = input[..., int((shape[-1] - 1) / 2)]
    else:
        output = (input[..., int(shape[-1] / 2 - 1)] + input[..., int(shape[-1] / 2)]) / 2.0
    return output

def repeated_median(y):
    eps = np.finfo(float).eps
    num_models = y.shape[1]
    total_num = y.shape[0]
    y = y.sort()[0]
    yyj = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyi = yyj.transpose(-1, -2)
    xx = torch.FloatTensor(range(num_models)).to(y.device)
    xxj = xx.repeat(total_num, num_models, 1)
    xxi = xxj.transpose(-1, -2) + eps

    diag = torch.Tensor([float("Inf")] * num_models).to(y.device)
    diag = torch.diag(diag).repeat(total_num, 1, 1)

    dividor = xxi - xxj + diag
    slopes = (yyi - yyj) / dividor + diag
    slopes, _ = slopes.sort()
    slopes = median_opt(slopes[:, :, :-1])
    slopes = median_opt(slopes)

    # get intercepts (intercept of median)
    yy_median = median_opt(y)
    xx_median = [(num_models - 1) / 2.0] * total_num
    xx_median = torch.Tensor(xx_median).to(y.device)
    intercepts = yy_median - slopes * xx_median

    return slopes, intercepts

def weighted_average(w_list, weights):
    w_avg = copy.deepcopy(w_list[0])
    weights = weights / weights.sum()
    assert len(weights) == len(w_list)
    for k in w_avg.keys():
        w_avg[k] = 0
        for i in range(0, len(w_list)):
            w_avg[k] += w_list[i][k] * weights[i]
        # w_avg[k] = torch.div(w_avg[k], len(w_list))
    return w_avg, weights

def do_dnc(grad_vec,niters=1,c=1,b=10000,m=2,n=10):
    """
    For all these datasets, we set niters, c, and b in Algorithm 2 to 1, 1, and 10,000, respectively
    """
    n = grad_vec.shape[0]
    d = grad_vec.shape[1]
    I_good = []
    for i in range(niters):
        r = torch.randperm(d)[:b]
        r = r.sort().values
        s_grad_vec = grad_vec[:,r] # n,b
        mu = s_grad_vec.mean(dim=0) #  1,b
        grad_c = s_grad_vec - mu
        svd_res = torch.linalg.svd(grad_c)
        v = svd_res.Vh[0] # top right singular eigenvector, (n)
        outlier_score = torch.Tensor([torch.dot(s_grad_vec[j]-mu, v).pow(2) for j in range(n)])
        i_val, i_good = outlier_score.sort()
        I = i_good[:n-c*m]
        I_good.append(set(I.tolist()))
    I_final = set.intersection(*I_good)
    I_final = torch.LongTensor(list(I_final))
    wvs = torch.zeros(n)
    for i in I_final:
        wvs[i] = 1
    return I_final, wvs, outlier_score

def foolsgold_wv_update(wv, grads, global_weight):
    """
    update global model with aggregated wv values
    
    """
    n_clients = grads.shape[0]
    
    model_weight_list = []
    for i in range(0, n_clients):
        if wv[i] != 0:
            current_weight = global_weight + wv[i] * grads[i]
            model_weight_list.append(current_weight)
    fools_gold_weight = torch.cat(model_weight_list).mean(0, keepdims=True)
    return fools_gold_weight.view(-1)