import argparse
import torch
from torch_geometric.data import HeteroData, Data
from torch_geometric.utils import add_self_loops, to_undirected
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from torch_geometric.nn import to_hetero
import os
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader, HGTLoader, RandomNodeLoader
from sklearn.metrics import precision_score, recall_score, f1_score ,accuracy_score, accuracy_score, classification_report, roc_auc_score, roc_curve, auc
import csv
from tqdm import trange
from utils.homogeneous_dataloader import load_homogeneous_cert_data, device_sharing_relationship, email_communication_relationship, user_hierarchical_relationship, none_homogeneous_relationship
from utils.heterogeneous_dataloader import load_heterogeneous_cert_data, none_heterogeneous_edges, process_sequential_edges, process_log2vec_edges
from utils.models import ResHybNet, EnhancedResHybNet, HetResHybnet
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
parser.add_argument('--method', type=str, choices=['undersampling', 'undersampling_new', 'undersampling_Reshybnet', 'one_month', 'oversampling'], default='undersampling', help='Sampling method')
parser.add_argument('--gnn_type', type=str, choices=['HGT', 'GAT', 'SAGE', 'all'], default='all', help='Type of GNN to use')

# Training arguments
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cpu', 'cuda'], help='Device to use for training')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
parser.add_argument('--round', type=int, default=10, help='Round of training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--cluster', action='store_true', help='Whether to use clustering')
parser.add_argument('--resume', action='store_true', help='Whether to resume training from a specific round')

parser.add_argument('--edge_type', type=str, choices=['none', 'sequential', 'log2vec', 'all'], default='all', help='Type of edge in the graph')
parser.add_argument('--num_feat', type=num_feat_all_or_int, default='all', choices=['all', 5, 15], help='Number of features to use')

# Parse initial arguments to determine conditional arguments
args, unknown = parser.parse_known_args()

# Add conditional arguments based on parsed values
if args.method == 'one_month':
    parser.add_argument('--year', type=int, default=2010, choices=[2010, 2011], help='Year to use for one month sampling')
    parser.add_argument('--month', type=int, default=1, help='Month to use for one month sampling')
elif args.method == 'oversampling':
    parser.add_argument('--oversample_path', type=str, default='./upsampling/email/adj_up_cert_upsampling_1743608921.4495988.pkl', help='Path to the oversampled data')

if args.resume:
    parser.add_argument('--resume_round', type=int, default=0, help='Round to resume training from')

# Final parsing of all arguments
args = parser.parse_args()

# args = {}
# args['data_path'] = "/fred/oz382/dataset/CERT/r4.2/rs_data"
# args['year'] = 2010
# args['month'] = 8
# args['edge_type'] = "all"
# args['num_feat'] = "all"
# args['gnn_type'] = "all"
# args['device'] = "cuda"
# args['num_epochs'] = 1000
# args['round'] = 10
# args['learning_rate'] = 0.001
# args['version'] = "r4.2"
# args['cluster'] = True
# args['method'] = "undersampling"
# args['resume'] = False
# # convert dict to object
# class Args:
#     def __init__(self, **entries):
#         self.__dict__.update(entries)

# args = Args(**args)

print(f"Starting training with the following parameters:")
print(f"Data Path: {args.data_path}")
print(f"Version: {args.version}")
print(f"Method: {args.method}")
if args.method == 'one_month':
    print(f"Year: {args.year}")
    print(f"Month: {args.month}")
elif args.method == 'oversampling':
    print(f"Oversample Path: {args.oversample_path}")
print(f"Edge Type: {args.edge_type}")
print(f"Number of Features: {args.num_feat}")
print(f"Cluster: {args.cluster}")
print(f"GNN Type: {args.gnn_type}")
print(f"Number of Epochs: {args.num_epochs}")
print(f"Round: {args.round}")
print(f"Learning Rate: {args.learning_rate}")
print(f"Resume Training: {args.resume}")
if args.resume:
    print(f"Resume Round: {args.resume_round}")

GNNS = []
if args.gnn_type == 'HGT' or args.gnn_type == 'all':
    GNNS.append('HGT')
if args.gnn_type == 'GAT' or args.gnn_type == 'all':
    GNNS.append('GAT')
if args.gnn_type == 'SAGE' or args.gnn_type == 'all':
    GNNS.append('SAGE')
device = torch.device(args.device)

all_result= './detection_result'
detect_model=f"{'Cluster' if args.cluster else 'NoCluster'}_HetGNN_compare_sequence_length_{args.round}round"
result_dir= os.path.join(all_result, "Heterogeneous", args.method, detect_model) 
os.makedirs(result_dir,exist_ok=True)

year = args.year if args.method == 'one_month' else None
month = args.month if args.method == 'one_month' else None

scaler = StandardScaler()

# Load data
if args.method != 'oversampling':
    filtered, activities, labels, users = load_heterogeneous_cert_data(args.data_path, args.method, args.version, year, month)

    conns = {}
    # Create relationship edges
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

conns_key = deepcopy(list(conns.keys()))

for conn_type in conns_key:
    conn = conns[conn_type]
    os.makedirs(os.path.join(result_dir, conn_type), exist_ok=True)

    # Build HeteroData
    graph = HeteroData()
    for ntype, df in activities.items():
        feats = df.to_numpy().astype(np.float32)
        feats = scaler.fit_transform(
            feats[:, :args.num_feat] if args.num_feat != 'all' else feats
        )
        graph[ntype].x = torch.from_numpy(feats)
        graph[ntype].y = torch.from_numpy(labels[ntype])
        graph[ntype].num_nodes = graph[ntype].x.size(0)
    for edge_key, idxs in conn.items():
        src, rel, dst = edge_key.split('_')
        graph[src, rel, dst].edge_index = torch.tensor(idxs).long()
    for ntype in graph.node_types:
        N = graph[ntype].num_nodes
        perm = torch.randperm(N)
        n_test = int(N * 0.2)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        mtr = torch.zeros(N, dtype=torch.bool)
        mte = torch.zeros(N, dtype=torch.bool)
        mtr[train_idx] = True
        mte[test_idx] = True
        graph[ntype].train_mask = mtr
        graph[ntype].test_mask  = mte
    ori_data = pad_hetero_features(graph)

    node_types, edge_types = ori_data.metadata()
    input_dims = {nt: ori_data[nt].x.size(1) for nt in node_types}
    output_dims = {nt: int(ori_data[nt].y.max().item()) + 1 for nt in node_types}

    for GNN in GNNS:
        perf_file = os.path.join(
            result_dir, conn_type, f'{CNN}_{GNN}_{args.num_feat}_{RESIDUAL}_result.csv'
        )

        for r in range(args.resume_round if args.resume else 0, args.round):
            model = HetResHybnet(
                metadata=(node_types, edge_types),
                input_dims=input_dims,
                output_dims=output_dims,
                cnn=CNN,
                gnn=GNN,
                residual=(RESIDUAL == 'YES')
            ).to(device)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.learning_rate, weight_decay=5e-4
            )
            best_model_dir = os.path.join(result_dir, conn_type, "early_stop_model")
            os.makedirs(best_model_dir, exist_ok=True)
            classification_report_path = os.path.join(
                result_dir, conn_type, "classification_report"
            )
            os.makedirs(classification_report_path, exist_ok=True)
            best_model_path = os.path.join(
                best_model_dir,
                f'{CNN}_{GNN}_{args.num_feat}_{conn_type}_{RESIDUAL}_{r}round_best.pt'
            )
            early_stopping = EarlyStopping(
                save_path=best_model_path,
                verbose=True,
                patience=40,
                delta=0.0001,
                metric='loss'
            )
            train_losses, test_losses = [], []

            if args.cluster:
                train_loader = RandomNodeLoader(ori_data, num_parts=32, shuffle=True)
                test_loader = RandomNodeLoader(ori_data, num_parts=10, shuffle=False)

            for epoch in range(args.num_epochs):
                model.train()
                if args.cluster:
                    epoch_loss = 0.0
                    all_train_preds = {ntype: [] for ntype in node_types}
                    all_train_labels = {ntype: [] for ntype in node_types}
                    for batch in train_loader:
                        data = batch.to(device)
                        optimizer.zero_grad()
                        out = model(data)
                        preds_dict = {ntype: logits.cpu().argmax(dim=1) for ntype, logits in out.items()}
                        y_dict = {ntype: data[ntype].y.cpu() for ntype in node_types}
                        mask_dict = {ntype: data[ntype].train_mask.cpu() for ntype in node_types}
                        train_loss = 0.0
                        valid_types_train = 0
                        for ntype in node_types:
                            mask_train = mask_dict[ntype]
                            if mask_train.sum().item() == 0:
                                continue
                            y_true = y_dict[ntype]
                            logits = out[ntype].cpu()
                            loss = F.nll_loss(logits[mask_train], y_true[mask_train])
                            if torch.isnan(loss):
                                print(f"Warning: NaN loss at training for node type {ntype}, epoch {epoch}. Skipping.")
                                continue
                            train_loss += loss
                            valid_types_train += 1
                            all_train_preds[ntype].append(preds_dict[ntype][mask_train])
                            all_train_labels[ntype].append(y_true[mask_train])
                        if valid_types_train > 0:
                            train_loss = train_loss / valid_types_train
                            train_loss.backward()
                            optimizer.step()
                            epoch_loss += train_loss.item()
                    if len(train_loader) > 0:
                        epoch_loss /= len(train_loader)
                    else:
                        epoch_loss = 0.0
                    train_losses.append(epoch_loss)

                    for ntype in node_types:
                        if all_train_labels[ntype]:
                            all_train_labels[ntype] = torch.cat(all_train_labels[ntype])
                            all_train_preds[ntype] = torch.cat(all_train_preds[ntype])
                        else:
                            all_train_labels[ntype] = torch.tensor([]).long()
                            all_train_preds[ntype] = torch.tensor([]).long()

                    with open(
                        f"{classification_report_path}/{conn_type}_CNN_{GNN}_{args.num_feat}_train.txt",
                        'a+'
                    ) as f:
                        f.write(f"Round {r} (Training), Epoch {epoch}, Loss: {epoch_loss:.4f}\n")
                        for ntype in node_types:
                            if all_train_labels[ntype].numel() > 0:
                                f.write(f"Node Type: {ntype}\n")
                                f.write(f"Report:\n{classification_report(all_train_labels[ntype], all_train_preds[ntype])}\n")

                    # Validation
                    model.eval()
                    with torch.no_grad():
                        test_loss_main = 0.0
                        for batch in test_loader:
                            data = batch.to(device)
                            out = model(data)
                            preds_dict = {ntype: logits.cpu().argmax(dim=1) for ntype, logits in out.items()}
                            y_dict = {ntype: data[ntype].y.cpu() for ntype in node_types}
                            mask_dict = {ntype: data[ntype].test_mask.cpu() for ntype in node_types}
                            test_loss = 0.0
                            valid_types_test = 0
                            for ntype in node_types:
                                mask_test = mask_dict[ntype]
                                if mask_test.sum().item() == 0:
                                    continue
                                y_true = y_dict[ntype]
                                logits = out[ntype].cpu()
                                loss = F.nll_loss(logits[mask_test], y_true[mask_test])
                                if torch.isnan(loss):
                                    print(f"Warning: NaN loss at validation for node type {ntype}, epoch {epoch}. Skipping.")
                                    continue
                                test_loss += loss.item()
                                valid_types_test += 1
                            if valid_types_test > 0:
                                test_loss /= valid_types_test
                            else:
                                test_loss = 0.0
                            print(f"Validation batch loss: {test_loss:.4f}")
                            test_loss_main += test_loss
                        if len(test_loader) > 0:
                            test_loss_main /= len(test_loader)
                        else:
                            test_loss_main = 0.0
                        test_losses.append(test_loss_main)
                        print(f"Epoch {epoch} Test Loss: {test_loss_main:.4f}")
                    early_stopping(test_loss_main, model)
                    if early_stopping.early_stop:
                        print("Early stopping at epoch:", epoch)
                        break

                else:
                    data = ori_data.to(device)
                    optimizer.zero_grad()
                    out = model(data)
                    preds_dict = {ntype: logits.cpu().argmax(dim=1) for ntype, logits in out.items()}
                    y_dict = {ntype: data[ntype].y.cpu() for ntype in node_types}
                    mask_dict = {ntype: data[ntype].train_mask.cpu() for ntype in node_types}
                    train_loss = 0.0
                    valid_types_train = 0
                    for ntype in node_types:
                        mask_train = mask_dict[ntype]
                        if mask_train.sum().item() == 0:
                            continue
                        y_true = y_dict[ntype]
                        logits = out[ntype].cpu()
                        loss = F.nll_loss(logits[mask_train], y_true[mask_train])
                        if torch.isnan(loss):
                            print(f"Warning: NaN loss at full‐graph training for node type {ntype}, epoch {epoch}. Skipping.")
                            continue
                        train_loss += loss
                        valid_types_train += 1
                    if valid_types_train > 0:
                        train_loss = train_loss / valid_types_train
                        train_loss.backward()
                        optimizer.step()
                        train_losses.append(train_loss.item())
                    else:
                        train_losses.append(0.0)

                    with open(
                        f"{classification_report_path}/{conn_type}_CNN_{GNN}_{args.num_feat}_train.txt",
                        'a+'
                    ) as f:
                        f.write(f"Round {r} (Training), Epoch {epoch}, Loss: {train_losses[-1]:.4f}\n")
                        for ntype in node_types:
                            mask_train = mask_dict[ntype]
                            if mask_train.sum().item() > 0:
                                f.write(f"Node Type: {ntype}\n")
                                f.write(f"Report:\n{classification_report(y_dict[ntype][mask_train], preds_dict[ntype][mask_train])}\n")

                    # Full‐graph Validation
                    model.eval()
                    with torch.no_grad():
                        test_loss = 0.0
                        valid_types_test = 0
                        for ntype in node_types:
                            mask_test = data[ntype].test_mask.cpu()
                            if mask_test.sum().item() == 0:
                                continue
                            y_true = y_dict[ntype]
                            logits = out[ntype].cpu()
                            loss = F.nll_loss(logits[mask_test], y_true[mask_test])
                            if torch.isnan(loss):
                                print(f"Warning: NaN test loss full‐graph for node type {ntype}. Skipping.")
                                continue
                            test_loss += loss.item()
                            valid_types_test += 1
                        if valid_types_test > 0:
                            test_loss /= valid_types_test
                        else:
                            test_loss = 0.0
                        test_losses.append(test_loss)

                    early_stopping(test_loss, model)
                    if early_stopping.early_stop:
                        print("Early stopping at epoch:", epoch)
                        break

            # Final evaluation after loading best model
            model.load_state_dict(torch.load(best_model_path))
            model.eval()

            if args.cluster:
                all_test_preds = {ntype: [] for ntype in node_types}
                all_test_labels = {ntype: [] for ntype in node_types}
                all_test_probs = {ntype: [] for ntype in node_types}
                val_loss_main = 0.0
                for batch in test_loader:
                    data = batch.to(device)
                    out = model(data)
                    probs_dict = {ntype: logits.cpu().exp()[:, 1] for ntype, logits in out.items()}
                    preds_dict = {ntype: logits.cpu().argmax(dim=1) for ntype, logits in out.items()}
                    y_dict = {ntype: data[ntype].y.cpu() for ntype in node_types}
                    mask_test_dict = {ntype: data[ntype].test_mask.cpu() for ntype in node_types}
                    val_loss = 0.0
                    valid_types_val = 0
                    for ntype in node_types:
                        mask_test = mask_test_dict[ntype]
                        if mask_test.sum().item() == 0:
                            continue
                        y_true = y_dict[ntype]
                        logits = out[ntype].cpu()
                        loss = F.nll_loss(logits[mask_test], y_true[mask_test])
                        if torch.isnan(loss):
                            print(f"Warning: NaN loss at final evaluation for node type {ntype}. Skipping.")
                            continue
                        val_loss += loss.item()
                        valid_types_val += 1
                        all_test_probs[ntype].append(probs_dict[ntype][mask_test])
                        all_test_preds[ntype].append(preds_dict[ntype][mask_test])
                        all_test_labels[ntype].append(y_true[mask_test])
                    if valid_types_val > 0:
                        val_loss /= valid_types_val
                    else:
                        val_loss = 0.0
                    val_loss_main += val_loss
                if len(test_loader) > 0:
                    val_loss_main /= len(test_loader)
                else:
                    val_loss_main = 0.0

                with open(
                    f"{classification_report_path}/{conn_type}_CNN_{GNN}_{args.num_feat}_validate.txt",
                    'a+'
                ) as f:
                    f.write(f"Round {r} (Validation), Epoch {epoch}, Loss: {val_loss_main:.4f}\n")
                    for ntype in node_types:
                        if all_test_labels[ntype]:
                            all_test_probs[ntype] = torch.cat(all_test_probs[ntype])
                            all_test_labels[ntype] = torch.cat(all_test_labels[ntype])
                            all_test_preds[ntype] = torch.cat(all_test_preds[ntype])
                        else:
                            all_test_probs[ntype] = torch.tensor([]).float()
                            all_test_labels[ntype] = torch.tensor([]).long()
                            all_test_preds[ntype] = torch.tensor([]).long()
                        if all_test_labels[ntype].numel() > 0:
                            acc = accuracy_score(all_test_labels[ntype], all_test_preds[ntype])
                            pre = precision_score(all_test_labels[ntype], all_test_preds[ntype], zero_division=0)
                            rec = recall_score(all_test_labels[ntype], all_test_preds[ntype], zero_division=0)
                            f1 = f1_score(all_test_labels[ntype], all_test_preds[ntype], zero_division=0)
                            fpr, tpr, _ = roc_curve(all_test_labels[ntype].detach().cpu().numpy(), all_test_probs[ntype].detach().cpu().numpy())
                            this_node_test_auc = auc(fpr, tpr)

                            plt.figure()
                            plt.plot(fpr, tpr, label=f'ROC curve (area = {this_node_test_auc:.2f})')
                            plt.plot([0, 1], [0,1], 'k--', label='Random guess')
                            plt.xlabel("False Positive Rate")
                            plt.ylabel("True Positive Rate")
                            plt.title(f"ROC Curve for Node Type: {ntype}")
                            plt.legend()
                            plt.savefig(os.path.join(result_dir, conn_type, f'ROC_Curve_{CNN}_{GNN}_{args.num_feat}_{RESIDUAL}_{ntype}_round{r}.png'))
                            plt.close()
                            
                            f.write(f"Node Type: {ntype}\n")
                            f.write(f"Report:\n{classification_report(all_test_labels[ntype], all_test_preds[ntype])}\n")
                            with open(perf_file, 'a', newline='') as pf:
                                writer = csv.writer(pf)
                                writer.writerow([
                                    f'{CNN}_{GNN}', f'feat_{args.num_feat}_{ntype}',
                                    r, acc, pre, rec, f1, this_node_test_auc, epoch
                                ])
            else:
                data = ori_data.to(device)
                out = model(data)
                probs_dict = {ntype: logits.cpu().exp()[:, 1] for ntype, logits in out.items()}
                preds_dict = {ntype: logits.cpu().argmax(dim=1) for ntype, logits in out.items()}
                y_dict = {ntype: data[ntype].y.cpu() for ntype in node_types}
                mask_dict = {ntype: data[ntype].test_mask.cpu() for ntype in node_types}
                val_loss = 0.0
                valid_types_val = 0
                all_test_probs = {ntype: [] for ntype in node_types}
                all_test_preds = {ntype: [] for ntype in node_types}
                all_test_labels = {ntype: [] for ntype in node_types}
                for ntype in node_types:
                    mask_test = mask_dict[ntype]
                    if mask_test.sum().item() == 0:
                        continue
                    y_true = y_dict[ntype]
                    logits = out[ntype].cpu()
                    loss = F.nll_loss(logits[mask_test], y_true[mask_test])
                    if torch.isnan(loss):
                        print(f"Warning: NaN loss at final full-graph evaluation for node type {ntype}. Skipping.")
                        continue
                    val_loss += loss.item()
                    valid_types_val += 1
                    all_test_probs[ntype].append(probs_dict[ntype][mask_test])
                    all_test_preds[ntype].append(preds_dict[ntype][mask_test])
                    all_test_labels[ntype].append(y_true[mask_test])
                if valid_types_val > 0:
                    val_loss /= valid_types_val
                else:
                    val_loss = 0.0
                with open(
                    f"{classification_report_path}/{conn_type}_CNN_{GNN}_{args.num_feat}_validate.txt",
                    'a+'
                ) as f:
                    f.write(f"Round {r} (Validation), Epoch {epoch}, Loss: {val_loss:.4f}\n")
                    for ntype in node_types:
                        if all_test_labels[ntype]:
                            all_test_probs[ntype] = torch.cat(all_test_probs[ntype])
                            all_test_labels[ntype] = torch.cat(all_test_labels[ntype])
                            all_test_preds[ntype] = torch.cat(all_test_preds[ntype])
                        else:
                            all_test_probs[ntype] = torch.tensor([]).float()
                            all_test_labels[ntype] = torch.tensor([]).long()
                            all_test_preds[ntype] = torch.tensor([]).long()
                        if all_test_labels[ntype].numel() > 0:
                            acc = accuracy_score(all_test_labels[ntype], all_test_preds[ntype])
                            pre = precision_score(all_test_labels[ntype], all_test_preds[ntype], zero_division=0)
                            rec = recall_score(all_test_labels[ntype], all_test_preds[ntype], zero_division=0)
                            f1 = f1_score(all_test_labels[ntype], all_test_preds[ntype], zero_division=0)
                            fpr, tpr, _ = roc_curve(all_test_labels[ntype].detach().cpu().numpy(), all_test_probs[ntype].detach().cpu().numpy())
                            this_node_test_auc = auc(fpr, tpr)

                            plt.figure()
                            plt.plot(fpr, tpr, label=f'ROC curve (area = {this_node_test_auc:.2f})')
                            plt.plot([0, 1], [0,1], 'k--', label='Random guess')
                            plt.xlabel("False Positive Rate")
                            plt.ylabel("True Positive Rate")
                            plt.title(f"ROC Curve for Node Type: {ntype}")
                            plt.legend()
                            plt.savefig(os.path.join(result_dir, conn_type, f'ROC_Curve_{CNN}_{GNN}_{args.num_feat}_{RESIDUAL}_{ntype}_round{r}.png'))
                            plt.close()

                            f.write(f"Node Type: {ntype}\n")
                            f.write(f"Report:\n{classification_report(all_test_labels[ntype], all_test_preds[ntype])}\n")
                            with open(perf_file, 'a', newline='') as pf:
                                writer = csv.writer(pf)
                                writer.writerow([
                                    f'{CNN}_{GNN}', f'feat_{args.num_feat}_{ntype}',
                                    r, acc, pre, rec, f1, this_node_test_auc, epoch
                                ])

            # Cleanup
            del data, model, train_loader, test_loader

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
    del conns[conn_type], ori_data
    gc.collect()