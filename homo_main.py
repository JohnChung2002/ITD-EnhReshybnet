import argparse
import torch
from torch_geometric.data import HeteroData, Data
from torch_geometric.utils import add_self_loops, to_undirected
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from torch_geometric.nn import to_hetero
import os
from torch_geometric.loader import ClusterData, ClusterLoader
from sklearn.metrics import precision_score, recall_score, f1_score ,accuracy_score, accuracy_score, classification_report, roc_auc_score, roc_curve, auc
import csv
from tqdm import trange
from utils.homogeneous_dataloader import load_homogeneous_cert_data, device_sharing_relationship, email_communication_relationship, user_hierarchical_relationship, none_homogeneous_relationship
from utils.heterogeneous_dataloader import load_heterogeneous_cert_data, none_heterogeneous_edges, process_sequential_edges, process_log2vec_edges
from utils.models import ResHybNet, EnhancedResHybNet
from utils.utility import EarlyStopping, calculate_minibatch_params, adj_to_edge_index, last_day_of_month, pad_hetero_features
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from copy import deepcopy
import gc

CNN='CNN'  # 'CNN' or ''
RESIDUAL='YES' # 'YES' or 'NO'

def num_feat_all_or_int(arg):
    try:
        return int(arg)  # try convert to int
    except ValueError:
        pass
    if arg == "all":
        return arg
    raise argparse.ArgumentTypeError("num_feat must be an int or 'all'")

parser = argparse.ArgumentParser(description='Graph Insider Threat Detection')

# Data arguments
parser.add_argument('--data_path', type=str, default='./data', help='Path to the dataset files')
parser.add_argument('--version', type=str, default='r4.2', help='Version of the dataset')
parser.add_argument('--graph_type', type=str, choices=['Heterogeneous', 'Homogeneous'], default='Heterogeneous', help='Type of graph')
parser.add_argument('--method', type=str, choices=['undersampling', 'undersampling_new', 'undersampling_Reshybnet', 'one_month', 'oversampling'], default='undersampling', help='Sampling method')

# Model arguments
parser.add_argument('--model_type', type=str, choices=['ResHybNet', 'EnhancedResHybNet'], default='EnhancedResHybNet', help='Type of model to use')
parser.add_argument('--gnn_type', type=str, choices=['GCN', 'GAT', 'SAGE', 'all'], default='all', help='Type of GNN to use')

# Training arguments
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cpu', 'cuda'], help='Device to use for training')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
parser.add_argument('--round', type=int, default=10, help='Round of training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--cluster', action='store_true', help='Whether to use clustering')
parser.add_argument('--resume', action='store_true', help='Whether to resume training from a specific round')

# Parse initial arguments to determine conditional arguments
args, unknown = parser.parse_known_args()

# Add conditional arguments based on parsed values
if args.method == 'one_month':
    parser.add_argument('--year', type=int, default=2010, choices=[2010, 2011], help='Year to use for one month sampling')
    parser.add_argument('--month', type=int, default=1, help='Month to use for one month sampling')
elif args.method == 'oversampling':
    parser.add_argument('--oversample_path', type=str, default='./upsampling/email/adj_up_cert_upsampling_1743608921.4495988.pkl', help='Path to the oversampled data')

if args.graph_type == 'Heterogeneous':
    parser.add_argument('--edge_type', type=str, choices=['none', 'sequential', 'log2vec', 'all'], default='all', help='Type of edge in the graph')
    parser.add_argument('--num_feat', type=num_feat_all_or_int, default='all', choices=['all', 5, 15], help='Number of features to use')
elif args.graph_type == 'Homogeneous':
    parser.add_argument('--edge_type', type=str, choices=['none', 'device', 'user', 'email', 'all'], default='all', help='Type of edge in the graph')
    parser.add_argument('--num_feat', type=num_feat_all_or_int, default='all', choices=['all', 5, 15, 25, 35, 45, 55], help='Number of features to use')

if args.resume:
    parser.add_argument('--resume_round', type=int, default=0, help='Round to resume training from')

# Final parsing of all arguments
args = parser.parse_args()

print(f"Starting training with the following parameters:")
print(f"Data Path: {args.data_path}")
print(f"Version: {args.version}")
print(f"Graph Type: {args.graph_type}")
print(f"Method: {args.method}")
if args.method == 'one_month':
    print(f"Year: {args.year}")
    print(f"Month: {args.month}")
elif args.method == 'oversampling':
    print(f"Oversample Path: {args.oversample_path}")
print(f"Edge Type: {args.edge_type}")
print(f"Number of Features: {args.num_feat}")
print(f"Cluster: {args.cluster}")
print(f"Model Type: {args.model_type}")
print(f"GNN Type: {args.gnn_type}")
print(f"Number of Epochs: {args.num_epochs}")
print(f"Round: {args.round}")
print(f"Learning Rate: {args.learning_rate}")
print(f"Resume Training: {args.resume}")
if args.resume:
    print(f"Resume Round: {args.resume_round}")

GNNS = []
if args.gnn_type == 'GCN' or args.gnn_type == 'all':
    GNNS.append('GCN')
if args.gnn_type == 'GAT' or args.gnn_type == 'all':
    GNNS.append('GAT')
if args.gnn_type == 'SAGE' or args.gnn_type == 'all':
    GNNS.append('SAGE')
device = torch.device(args.device)

all_result= './detection_result'
detect_model=f"{'Cluster' if args.cluster else 'NoCluster'}{args.model_type}_compare_sequence_length_{args.round}round"
result_dir= os.path.join(all_result, args.graph_type, args.method, detect_model) 
os.makedirs(result_dir,exist_ok=True)

year = args.year if args.method == 'one_month' else None
month = args.month if args.method == 'one_month' else None

scaler = StandardScaler()

# Load data
if args.method != 'oversampling':
    if args.graph_type == 'Heterogeneous':
        filtered, activities, labels, users = load_heterogeneous_cert_data(args.data_path, args.method, args.version, year, month)
    elif args.graph_type == 'Homogeneous':
        filtered, users, email, pcs = load_homogeneous_cert_data(args.data_path, args.method, args.version, year, month)

    conns = {}
    # Create relationship edges
    if args.graph_type == 'Heterogeneous':
        if args.method == 'one_month':
            #get start date of a given month
            start_date = datetime(year, month, 1) 
            #get last date of a given month 31 or 30 or 28 or 29
            end_date = last_day_of_month(start_date)
        else:
            min_date = ""
            max_date = ""
            for key in activities.keys():
                activities[key]['key_date'] = activities[key]['timestamp'].apply(lambda ts: str(datetime.fromtimestamp(ts).date()))
                min_date = min(min_date, activities[key]['key_date'].min()) if min_date != "" else activities[key]['key_date'].min()
                max_date = max(max_date, activities[key]['key_date'].max()) if max_date != "" else activities[key]['key_date'].max()
            start_date = datetime.strptime(min_date, '%Y-%m-%d')
            end_date = datetime.strptime(max_date, '%Y-%m-%d')
            for key in activities.keys():
                activities[key].drop(columns=['key_date'], inplace=True)        
        if args.edge_type == 'none' or args.edge_type == 'all':
            conns['none'] = none_heterogeneous_edges(activities)
        if args.edge_type == 'sequential' or args.edge_type == 'all':
            conns['sequential'] = process_sequential_edges(filtered, start_date, end_date)
        if args.edge_type == 'log2vec' or args.edge_type == 'all':
            conns['log2vec'] = process_log2vec_edges(filtered, start_date, end_date)
    elif args.graph_type == 'Homogeneous':
        if args.edge_type == 'none' or args.edge_type == 'all':
            conns['none'] = none_homogeneous_relationship(filtered)
        if args.edge_type == 'user' or args.edge_type == 'all':
            conns['user'] = user_hierarchical_relationship(filtered, users)
        if args.edge_type == 'device' or args.edge_type == 'all':
            conns['device'] = device_sharing_relationship(filtered, pcs)
        if args.edge_type == 'email' or args.edge_type == 'all':
            conns['email'] = email_communication_relationship(filtered, email, users)
else:
    if "recon" in args.oversample_path:
        with open(args.oversample_path, "rb") as f:
            embed, labels, idx_train, adj = pickle.load(f)
        features = scaler.fit_transform(embed.detach())
    else:
        with open(args.oversample_path, "rb") as f:
            adj, features, labels, idx_train = pickle.load(f)
            adj = adj.to_dense()
        features = scaler.fit_transform(features)
        
    if args.num_feat == 'all':
        principal_components = features
    else:
        pca = PCA(n_components=args.num_feat)
        principal_components = pca.fit_transform(features)
    edge_index = adj_to_edge_index(adj)
    ori_data = Data(x=torch.FloatTensor(principal_components), edge_index=edge_index, y=torch.LongTensor(labels))

    del features
    del adj
    del principal_components
    del labels

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    conns = {}
    pathlist = os.path.normpath(args.oversample_path).split(os.path.sep)
    connection_type = f"{pathlist[-3]}_{pathlist[-2]}"
    conns[connection_type] = []

conns_key = list(conns.keys())
for conn_type in conns_key:
    os.makedirs(os.path.join(result_dir, conn_type),exist_ok=True)
    if args.graph_type == 'Heterogeneous':
        if args.method != 'oversampling':
            principal_components = {}
            for key in activities:
                principal_components[key] = torch.from_numpy(activities[key].to_numpy().astype(np.float32))
                principal_components[key] = scaler.fit_transform(principal_components[key])
                if args.num_feat != 'all':
                    pca = PCA(n_components=args.num_feat)
                    principal_components[key] = pca.fit_transform(principal_components[key])

            conn = conns[conn_type]
            graph = HeteroData()
            for key in principal_components:
                graph[key].x = torch.from_numpy(principal_components[key])
                graph[key].y = torch.from_numpy(labels[key])
                graph[key].num_nodes = len(principal_components[key])
            for edge_key in conn:
                node1, relationship, node2 = edge_key.split("_")
                conn[edge_key] = torch.tensor(conn[edge_key])
                graph[node1, relationship, node2].edge_index = conn[edge_key]
            for key in graph.keys():
                if 'edge_index' in graph[key]:
                    graph[key].edge_index = add_self_loops(graph[key].edge_index, num_nodes=graph[key].num_nodes)[0]
            del graph['edge_index']
            del graph['num_nodes']
            del graph['x']
            del graph['y']

            # max_feat_num = max([graph[node_type].x.size(1) for node_type in graph.node_types])
            # graph = pad_hetero_features(graph)

            # ori_data = graph
            ori_data = graph.to_homogeneous()
            ori_data.edge_index = ori_data.edge_index.long()
            # Count directed edges in homo_data.edge_index
            src = ori_data.edge_index[0]
            dst = ori_data.edge_index[1]

            # Create a set of edges for quick lookup
            edges = set(zip(src.tolist(), dst.tolist()))

            directed = [[], []]

            # Count edges that do not have a reverse counterpart
            for edge in edges:
                if (edge[1], edge[0]) not in edges:
                    directed[0].append(edge[1])
                    directed[1].append(edge[0])

            directed = torch.LongTensor(directed)

            ori_data.edge_index = torch.cat(
                (ori_data.edge_index, directed), dim=1
            )
            ori_data.x = ori_data.x.float()
            del directed
            del edges 
            del conns[conn_type]

            gc.collect()

    elif args.graph_type == 'Homogeneous':
        if args.method != 'oversampling':
            filtered_df = filtered.drop(columns=['key_date'])

            features = filtered_df.drop(['role', 'b_unit', 'f_unit', 'dept', 'team', 'ITAdmin', 'O', 'C', 'E', 'A', 'N', 'insider'], axis=1).to_numpy()
            features = torch.from_numpy(features).float()

            conn = torch.tensor(conns[conn_type], dtype=torch.long)

            labels = filtered_df['insider'].to_numpy()
            labels = torch.from_numpy(labels).long()

            features_scaled = scaler.fit_transform(features)
            if args.num_feat == 'all':
                principal_components = features_scaled
            else:
                pca = PCA(n_components=args.num_feat)
                principal_components = pca.fit_transform(features_scaled)

            principal_components = torch.FloatTensor(principal_components)

            ori_data = Data(x=principal_components, edge_index=conn, y=labels)

            del features
            del features_scaled
            del filtered_df
            del conns[conn_type]

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    for GNN in GNNS:
        perform_file=os.path.join(result_dir, conn_type, f'{CNN}_{GNN}_{args.num_feat}_{RESIDUAL}_result.csv')
        plot_file=perform_file.replace('results.csv','plot.npy')

        if args.method == 'undersampling_ResHybnet' and args.graph_type == 'Homogeneous':
            ori_data.train_mask = torch.zeros(ori_data.num_nodes, dtype=torch.uint8)
            ori_data.train_mask[:ori_data.num_nodes - 572] = 1                  
            ori_data.test_mask = torch.zeros(ori_data.num_nodes, dtype=torch.uint8)
            ori_data.test_mask[ori_data.num_nodes - 572:] = 1
        else:
            num_insiders = 0
            num_non_insiders = 0
            for label in ori_data.y:
                if label == 1:
                    num_insiders += 1
                else:
                    num_non_insiders += 1
            total_samples = num_insiders + num_non_insiders

            # Define 80/20 split
            train_ratio = 0.8
            test_ratio = 0.2

            # Calculate number of train and test samples
            num_insider_test = int(num_insiders * test_ratio)  # 20% of insiders
            num_non_insider_test = int(num_non_insiders * test_ratio)  # 20% of non-insiders

            num_insider_train = num_insiders - num_insider_test  # Remaining 80% insiders
            num_non_insider_train = num_non_insiders - num_non_insider_test  # Remaining 80% non-insiders

            # Create indices
            insider_indices = np.arange(num_insiders)  # First `num_insiders` are insiders
            non_insider_indices = np.arange(num_insiders, num_insiders + num_non_insiders)  # Remaining are non-insiders

            # Randomly sample test indices
            insider_test_indices = np.random.choice(insider_indices, num_insider_test, replace=False)
            non_insider_test_indices = np.random.choice(non_insider_indices, num_non_insider_test, replace=False)

            # Combine test indices
            test_indices = np.concatenate((insider_test_indices, non_insider_test_indices))

            # Create masks
            ori_data.train_mask = torch.zeros(total_samples, dtype=torch.uint8)
            ori_data.test_mask = torch.zeros(total_samples, dtype=torch.uint8)

            # Assign train-test split
            ori_data.test_mask[test_indices] = 1
            ori_data.train_mask[~ori_data.test_mask.bool()] = 1

            del test_indices
            del insider_indices
            del non_insider_indices

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # if args.resume:
        start_round = args.resume_round if args.resume else 0

        if args.graph_type == 'Heterogeneous':
            unique_node_types = ori_data.node_type.unique().tolist()
            node_type_mapping = dict(zip(unique_node_types, graph.node_types))
        
        for r in trange(start_round, args.round):
            train_loss_s = []
            test_loss_s = []

            data = deepcopy(ori_data)
            
            node_feat_num = data.num_node_features
            if args.model_type == 'EnhancedResHybNet':
                # node_feat_num = max_feat_num if args.graph_type == 'Heterogeneous' else data.num_node_features
                # node_feat_num = data.num_node_features
                model = EnhancedResHybNet(
                    input_dim=node_feat_num, 
                    output_dim=node_feat_num,
                    cnn=CNN,
                    gnn=GNN,
                    residual=RESIDUAL,
                    use_node2vec=True, 
                    use_pos_enc=True, 
                    num_pos_enc=8
                ).to(device)
                # if args.graph_type == 'Heterogeneous':
                #     model = to_hetero(model, data.metadata(), aggr='mean')
                model.setup_node2vec(data.edge_index, num_nodes=data.num_nodes, device=device)
                data = model.add_positional_encoding(data)     
            elif args.model_type == 'ResHybNet':
                # node_feat_num = data.num_node_features
                model = ResHybNet(input_dim=node_feat_num, 
                        output_dim=node_feat_num,
                        cnn=CNN, gnn=GNN, residual=RESIDUAL).to(device)
                # if args.graph_type == 'Heterogeneous':
                #     model = to_hetero(model, data.metadata(), aggr='mean')
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)

            best_model_path = os.path.join(result_dir, conn_type, "early_stop_model")
            os.makedirs(best_model_path, exist_ok=True)
            classification_report_path = os.path.join(result_dir, conn_type, "classification_report")
            os.makedirs(classification_report_path, exist_ok=True)   
            best_model_path = os.path.join(best_model_path, f'{CNN}_{GNN}_{args.num_feat}_{conn_type}_{RESIDUAL}_{r}round_best.pt')
            early_stopping = EarlyStopping(save_path=best_model_path, verbose=True, patience=40, delta=0.0001, metric='loss')

            if args.cluster:
                num_parts, batch_size, estimated_nodes = calculate_minibatch_params(data.num_nodes, desired_nodes=(4096 if args.graph_type == 'Heterogeneous' else 2048))

                print(f"Training model for round {r} with {args.num_feat} features")
                # Set up mini-batching for training and validation
                train_cluster_data = ClusterData(data, num_parts=num_parts)
                train_loader = ClusterLoader(train_cluster_data, batch_size=batch_size, shuffle=True)
                
                # For validation, use a separate loader (without shuffling)
                test_cluster_data = ClusterData(data, num_parts=num_parts)
                test_loader = ClusterLoader(test_cluster_data, batch_size=batch_size, shuffle=False)
            
            # Training loop with mini-batches for both training and validation
            for epoch in range(args.num_epochs):
                if args.cluster:
                    model.train()
                    epoch_loss = 0
                    all_train_preds = []
                    all_train_labels = []

                    for batch in train_loader:
                        # Only move the current batch to the GPU
                        batch = batch.to(device)
                        optimizer.zero_grad()
                        out = model(batch)
                        # Compute loss on training nodes in the batch
                        loss = F.nll_loss(out[batch.train_mask.bool()], batch.y[batch.train_mask.bool()].long())
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        _, out1 = out.max(dim=1)
                        preds_batch = torch.masked_select(out1, batch.train_mask.bool()).tolist()
                        labels_batch = batch.y[batch.train_mask.bool()].tolist()
                        all_train_preds.extend(preds_batch)
                        all_train_labels.extend(labels_batch)
                    epoch_loss /= len(train_loader)
                    train_loss_s.append(epoch_loss)
                    with open(f"{classification_report_path}/{conn_type}_CNN_{GNN}_{args.num_feat}_train.txt", 'a+') as f:
                        f.write(f"Round {r} (Training), Epoch {epoch}, Loss: {epoch_loss}\n")
                        f.write(f"Report: \n{classification_report(all_train_labels, all_train_preds)}")
                    
                    # Validation mini-batching
                    model.eval()
                    test_loss_total = 0.0
                    with torch.no_grad():
                        for batch in test_loader:
                            batch = batch.to(device)
                            out = model(batch)
                            # Compute loss only on nodes that are in the test set
                            loss_batch = F.nll_loss(out[batch.test_mask.bool()], batch.y[batch.test_mask.bool()].long())
                            test_loss_total += loss_batch.item()
                            _, out1 = out.max(dim=1)
                            preds_batch = torch.masked_select(out1, batch.test_mask.bool()).tolist()
                            labels_batch = batch.y[batch.test_mask.bool()].tolist()
                    test_loss = test_loss_total / len(test_loader)
                    test_loss_s.append(test_loss)
                else:
                    data = data.to(device)
                    model.train()
                    optimizer.zero_grad()
                    # Get output
                    out = model(data)
                    _, out1 = out.max(dim=1)
                    pred_y = torch.masked_select(out1, data.train_mask.bool()).tolist()
                    true_y = data.y[data.train_mask.bool()].tolist()
                    loss = F.nll_loss(out[data.train_mask.bool()], data.y[data.train_mask.bool()].long())
                    loss.backward()
                    optimizer.step()
                    train_loss_s.append(loss.item())

                    # if args.graph_type == 'Heterogeneous':
                    # else:
                    with open(f"{classification_report_path}/{conn_type}_CNN_{GNN}_{args.num_feat}_train.txt", 'a+') as f:
                        f.write(f"Round {r} (Training), Epoch {epoch}, Loss: {loss.item()}\n")
                        f.write(f"Report: \n{classification_report(true_y, pred_y)}")
                
                    model.eval()
                    out = model(data)
                    _, out1 = out.max(dim=1)
                    pred_y = torch.masked_select(out1, data.test_mask.bool()).tolist()
                    true_y = data.y[data.test_mask.bool()].tolist()
                    test_loss = F.nll_loss(out[data.test_mask.bool()], data.y[data.test_mask.bool()].long())
                    test_loss_s.append(test_loss.item())
                
                # Early stopping check based on aggregated validation loss
                early_stopping(test_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping at epoch:", epoch)
                    break
                
            early_stopping.draw_trend(train_loss_s, test_loss_s, graph_type=args.graph_type, method=args.method, detect_model=detect_model, connection=conn_type, perform_name=f'{CNN}_{GNN}_{args.num_feat}_{RESIDUAL}_{r}round')
            
            # Load best model and perform final evaluation on validation data via mini-batches
            model.load_state_dict(torch.load(best_model_path))
            model.eval()

            with torch.no_grad():

                if args.cluster:
                    if args.graph_type == 'Heterogeneous':
                        final_loss_total = 0.0
                        final_preds = {}
                        final_labels = {}
                        final_probs = {}
                        
                        for batch in test_loader:
                            batch = batch.to(device)
                            out = model(batch)
                            for key in node_type_mapping:
                                node_mask = (batch.node_type == key) & batch.test_mask.bool()
                                final_probs.setdefault(key, []).extend(out.exp()[:, 1][node_mask].tolist())
                                loss_batch = F.nll_loss(out[node_mask], batch.y[node_mask].long())
                                final_loss_total += loss_batch.item()
                                _, out1 = out.max(dim=1)
                                preds_batch = torch.masked_select(out1, node_mask).tolist()
                                labels_batch = batch.y[node_mask].tolist()
                                final_preds.setdefault(key, []).extend(preds_batch)
                                final_labels.setdefault(key, []).extend(labels_batch)
                        final_loss = final_loss_total / len(test_loader)
                    else:
                        final_loss_total = 0.0
                        final_preds = []
                        final_labels = []
                        final_probs = []
                        
                        for batch in test_loader:
                            batch = batch.to(device)
                            out = model(batch)
                            final_probs.extend(out.exp()[:, 1][batch.test_mask.bool()].tolist())
                            loss_batch = F.nll_loss(out[batch.test_mask.bool()], batch.y[batch.test_mask.bool()].long())
                            final_loss_total += loss_batch.item()
                            _, out1 = out.max(dim=1)
                            preds_batch = torch.masked_select(out1, batch.test_mask.bool()).tolist()
                            labels_batch = batch.y[batch.test_mask.bool()].tolist()
                            final_preds.extend(preds_batch)
                            final_labels.extend(labels_batch)
                        final_loss = final_loss_total / len(test_loader)
                else:
                    if args.graph_type == 'Heterogeneous':
                        out = model(data)
                        # mask to get only test_node and nodetype
                        final_probs = {}
                        final_preds = {}
                        final_labels = {}
                        final_loss = 0.0
                        for key in node_type_mapping:
                            node_mask = (data.node_type == key) & data.test_mask.bool()
                            final_probs[key] = out.exp()[:, 1][node_mask].tolist()
                            _, out1 = out.max(dim=1)
                            final_preds[key] = torch.masked_select(out1, node_mask).tolist()
                            final_labels[key] = data.y[node_mask].tolist()
                            final_loss += F.nll_loss(out[node_mask], data.y[node_mask].long()).item() 
                        final_loss /= len(node_type_mapping)
                    else:
                        out = model(data)
                        final_probs = out.exp()[:, 1][data.test_mask.bool()].tolist()
                        _, out1 = out.max(dim=1)
                        final_preds = torch.masked_select(out1, data.test_mask.bool()).tolist()
                        final_labels = data.y[data.test_mask.bool()].tolist()
                        final_loss = F.nll_loss(out[data.test_mask.bool()], data.y[data.test_mask.bool()].long()).item()
                
            if args.graph_type == 'Heterogeneous':
                with open(f"{classification_report_path}/{conn_type}_CNN_{GNN}_{args.num_feat}_validate.txt", 'a+') as file1:
                    file1.write(f"Round {r} (Validation), Epoch {epoch}, Loss {final_loss}\n")
                    for key in node_type_mapping:
                        file1.write(f"Node type: {node_type_mapping[key]}\n")
                        file1.write(f"Report: \n{classification_report(final_labels[key], final_preds[key])}\n")
                        this_node_acc = accuracy_score(final_labels[key], final_preds[key])
                        this_node_pre = precision_score(final_labels[key], final_preds[key])
                        this_node_rec = recall_score(final_labels[key], final_preds[key])
                        this_node_f1 = f1_score(final_labels[key], final_preds[key])
                        fpr, tpr, _ = roc_curve(final_labels[key], final_probs[key])
                        this_node_auc = auc(fpr, tpr)

                        # plot ROC curve
                        plt.figure()
                        plt.plot(fpr, tpr, label=f'ROC curve (area = {this_node_auc:.2f})')
                        plt.plot([0, 1], [0,1], 'k--', label='Random guess')
                        plt.xlabel("False Positive Rate")
                        plt.ylabel("True Positive Rate")
                        plt.title(f"ROC Curve for Node Type: {node_type_mapping[key]}")
                        plt.legend()
                        plt.savefig(os.path.join(result_dir, conn_type, f'ROC_Curve_{CNN}_{GNN}_{args.num_feat}_{RESIDUAL}_{node_type_mapping[key]}_round{r}.png'))
                        plt.close()

                        print(f'Best model testing performance for round {r}, node type {node_type_mapping[key]}:', 
                                this_node_acc, this_node_pre, this_node_rec, this_node_f1, this_node_auc)
                        with open(perform_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            my_list = [f'{CNN}_{GNN}', f'feat_{args.num_feat}_{node_type_mapping[key]}', r, this_node_acc, this_node_pre, this_node_rec, this_node_f1, this_node_auc, epoch]
                            writer.writerow(my_list)
            else:
                test_acc = accuracy_score(final_labels, final_preds)
                test_pre = precision_score(final_labels, final_preds)
                test_rec = recall_score(final_labels, final_preds)
                test_f1 = f1_score(final_labels, final_preds)
                test_auc = roc_auc_score(final_labels, final_preds)

                print(f'Best model testing performance for round {r}:', test_acc, test_pre, test_rec, test_f1, test_auc)
                
                with open(f"{classification_report_path}/{conn_type}_CNN_{GNN}_{args.num_feat}_validate.txt", 'a+') as f:
                    f.write(f"Round {r} (Validation), Epoch {epoch}, Loss {final_loss}\n")
                    f.write(f"Report: \n{classification_report(final_labels, final_preds)}")
                
                with open(perform_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    my_list = [f'{CNN}_{GNN}', f'feat_{args.num_feat}', r, test_acc, test_pre, test_rec, test_f1, test_auc, epoch]
                    writer.writerow(my_list)

            print('Finished round:', r)

            if args.cluster:
                del train_cluster_data
                del train_loader
                del test_cluster_data
                del test_loader
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()