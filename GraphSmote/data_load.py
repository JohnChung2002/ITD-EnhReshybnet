import argparse
import scipy.sparse as sp
import numpy as np
import torch
import ipdb
from scipy.io import loadmat
import utils
from collections import defaultdict
import pickle
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import itertools
from tqdm import tqdm

IMBALANCE_THRESH = 101

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target', type=int, default=4)    
    parser.add_argument('--k', type = int, default = 5)
    if hasattr(Trainer, 'add_args'):
        Trainer.add_args(parser)
    

    return parser

def load_data(path="data/cora/", dataset="cora"):#modified from code: pygcn
    """Load citation network dataset (cora only for now)"""
    #input: idx_features_labels, adj
    #idx,labels are not required to be processed in advance
    #adj: save in the form of edges. idx1 idx2 
    #output: adj, features, labels are all torch.tensor, in the dense form
    #-------------------------------------------------------

    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = idx_features_labels[:, -1]
    set_labels = set(labels)
    classes_dict = {c: np.arange(len(set_labels))[i] for i, c in enumerate(set_labels)}
    classes_dict = {'Neural_Networks': 0, 'Reinforcement_Learning': 1, 'Probabilistic_Methods': 2, 'Case_Based': 3, 'Theory': 4, 'Rule_Learning': 5, 'Genetic_Algorithms': 6}

    #ipdb.set_trace()
    labels = np.array(list(map(classes_dict.get, labels)))

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    utils.print_edges_num(adj.todense(), labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    #adj = torch.FloatTensor(np.array(adj.todense()))

    return adj, features, labels


def Extract_graph(edgelist, fake_node, node_num):
    
    node_list = range(node_num+1)[1:]
    node_set = set(node_list)
    adj_1 = sp.coo_matrix((np.ones(len(edgelist)), (edgelist[:, 0], edgelist[:, 1])), shape=(edgelist.max()+1, edgelist.max()+1), dtype=np.float32)
    adj_1 = adj_1 + adj_1.T.multiply(adj_1.T > adj_1) - adj_1.multiply(adj_1.T > adj_1)
    adj_csr = adj_1.tocsr()
    for i in np.arange(node_num):
        for j in adj_csr[i].nonzero()[1]:
            node_set.add(j)

    node_set_2 = node_set
    '''
    node_set_2 = set(node_list)
    for i in node_set:
        for j in adj_csr[i].nonzero()[1]:
            node_set_2.add(j)
    '''
    node_list = np.array(list(node_set_2))
    node_list = np.sort(node_list)
    adj_new = adj_csr[node_list,:]

    node_mapping = dict(zip(node_list, range(0, len(node_list), 1)))

    edge_list = []
    for i in range(len(node_list)):
        for j in adj_new[i].nonzero()[1]:
            if j in node_list:
                edge_list.append([i, node_mapping[j]])

    edge_list = np.array(edge_list)
    #adj_coo_new = sp.coo_matrix((np.ones(len(edge_list)), (edge_list[:,0], edge_list[:,1])), shape=(len(node_list), len(node_list)), dtype=np.float32)

    label_new = np.array(list(map(lambda x: 1 if x in fake_node else 0, node_list)))
    np.savetxt('data/twitter/sub_twitter_edges', edge_list,fmt='%d')
    np.savetxt('data/twitter/sub_twitter_labels', label_new,fmt='%d')

    return

def load_data_twitter():
    adj_path = 'data/twitter/twitter.csv'
    fake_id_path = 'data/twitter/twitter_fake_ids.csv'

    adj = np.loadtxt(adj_path, delimiter=',', skiprows=1)#(total: 16011444 edges, 5384162 nodes)
    adj = adj.astype(int)
    adj = np.array(adj,dtype=int)
    fake_node = np.genfromtxt(fake_id_path, delimiter=',',skip_header=1, usecols=(0), dtype=int)#(12437)
    
    #'''#using broad walk
    if False:
        Extract_graph(adj, fake_node, node_num=1000)

    #'''

    '''generated edgelist for deepwalk for embedding
    np.savetxt('data/twitter/twitter_edges', adj,fmt='%d')
    '''

    #process adj:
    adj[adj>50000] = 0 #save top 50000 node, start from 1
    adj = sp.coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(adj.max()+1, adj.max()+1), dtype=np.float32)
    adj = np.array(adj.todense())
    adj = adj[1:, 1:]
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj.tocoo()

    fake_node = np.sort(fake_node)
    fake_node = fake_node[fake_node<=50000]
    fake_id = fake_node-1

    #process label:
    labels = np.zeros((50000,),dtype=int)
    labels[fake_id] = 1


    #filtering out outliers:
    node_degree = adj.sum(axis=1)
    chosen_idx = np.arange(50000)[node_degree>=4]
    ipdb.set_trace()


    #embed need to be read sequentially, due to the size
    embed = np.genfromtxt('data/twitter/twitter.embeddings_64', max_rows=50000)
    feature = np.zeros((embed.shape[0],embed.shape[1]-1))
    feature[embed[:,0].astype(int),:] = embed[:,1:]
    features = normalize(feature)

    adj = adj[chosen_idx,:][:,chosen_idx]     #shape:
    labels = labels[chosen_idx]     #shape:
    features = features[chosen_idx]

    

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    utils.print_edges_num(adj.todense(), labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels

def load_sub_data_twitter():
    adj_path = 'data/twitter/sub_twitter_edges'
    fake_id_path = 'data/twitter/sub_twitter_labels'

    adj = np.loadtxt(adj_path, delimiter=' ', dtype=int)#
    adj = np.array(adj,dtype=int)
    labels = np.genfromtxt(fake_id_path, dtype=int)#(63167)
    
    #process adj:
    adj = sp.coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(adj.max()+1, adj.max()+1), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    #filtering out outliers:
    node_degree = np.array(adj.sum(axis=1)).reshape(-1)
    chosen_idx = np.arange(adj.shape[0])[node_degree>=4]#44982 nodes were left

    #embed need to be read sequentially, due to the size
    embed = np.genfromtxt('data/twitter/sub_node_embedding_64', skip_header=1)
    feature = np.zeros((embed.shape[0],embed.shape[1]-1))
    feature[embed[:,0].astype(int),:] = embed[:,1:]
    features = normalize(feature)

    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(labels)

    utils.print_edges_num(adj.todense(), labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels

def load_data_Blog():#
    #--------------------
    #
    #--------------------
    mat = loadmat('data/BlogCatalog/blogcatalog.mat')
    adj = mat['network']
    label = mat['group']

    embed = np.loadtxt('data/BlogCatalog/blogcatalog.embeddings_64')
    feature = np.zeros((embed.shape[0],embed.shape[1]-1))
    feature[embed[:,0].astype(int),:] = embed[:,1:]

    features = normalize(feature)
    labels = np.array(label.todense().argmax(axis=1)).squeeze()

    labels[labels>16] = labels[labels>16]-1

    print("change labels order, imbalanced classes to the end.")
    #ipdb.set_trace()
    labels = refine_label_order(labels)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    #adj = torch.FloatTensor(np.array(adj.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels


def none_homogeneous_relationship(df):
    # Create a list of empty lists for src and dst
    connections = [
        [],  # src
        []   # dst
    ]

    # Iterate over the DataFrame rows
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Append the same index to both src and dst
        connections[0].append(index)
        connections[1].append(index)

    return connections

# Function to create supervisor-to-user and intra-user connections
def user_hierarchical_relationship(df, users):
    # Precompute a mapping from each user (df['user'] value) to a list of df indices (user-days)
    # df['user'] is assumed to store the users' index from the users DataFrame.
    user_day_map = df.groupby('user').groups  # dict: user index -> Index of df rows

    # Create a mapping from user_id (string) to its index (i.e. row label in users)
    user_to_index = {row['user_id']: idx for idx, row in users.iterrows()}

    # Group subordinate users by supervisor. 
    # This creates a dictionary: supervisor -> list of subordinate user_ids.
    sup_to_userids = users.groupby('sup')['user_id'].apply(list).to_dict()

    connections = [[], []]

    # Process hierarchical edges for each supervisor.
    for supervisor, subordinate_ids in tqdm(sup_to_userids.items(), desc="Processing supervisors"):
        # Skip if supervisor is NaN or not in our user mapping.
        if pd.isna(supervisor) or supervisor not in user_to_index:
            continue

        supervisor_index = user_to_index[supervisor]
        supervisor_days = user_day_map.get(supervisor_index, [])

        if isinstance(supervisor_days, pd.DataFrame):
            if supervisor_days.empty:
                continue

        if isinstance(supervisor_days, list):
            if not supervisor_days:
                continue

        # Convert subordinate user_ids to their corresponding indices (filtering out missing ones)
        subordinate_indexes = [user_to_index[uid] for uid in subordinate_ids if uid in user_to_index]

        # 1. Add edges from supervisor days to each subordinate's days.
        for subordinate in subordinate_indexes:
            subordinate_days = user_day_map.get(subordinate, [])
            for src, dst in itertools.product(supervisor_days, subordinate_days):
                connections[0].append(src)
                connections[1].append(dst)
                connections[0].append(dst)
                connections[1].append(src)

        # 2. Add edges among subordinate users (all pairs of subordinate user-days)
        for u1, u2 in itertools.combinations(subordinate_indexes, 2):
            days_u1 = user_day_map.get(u1, [])
            days_u2 = user_day_map.get(u2, [])
            for src, dst in itertools.product(days_u1, days_u2):
                connections[0].append(src)
                connections[1].append(dst)
                connections[0].append(dst)
                connections[1].append(src)

    # Process self edges (within each user): connect all user-days from the same user.
    for user_id, user_index in tqdm(user_to_index.items(), desc="Processing self connections"):
        user_days = user_day_map.get(user_index, [])
        for src, dst in itertools.combinations(user_days, 2):
            connections[0].append(src)
            connections[1].append(dst)
            connections[0].append(dst)
            connections[1].append(src)

    return connections

def merge_all_dates_for_each_user(df):
    # Custom function to merge lists in the 'to' column
    def merge_to_lists(series):
        merged = []
        for item in series:
            # Make sure each item is a list
            if not isinstance(item, list):
                item = [item]
            merged.extend(item)
        # Remove duplicates while preserving order
        return list(dict.fromkeys(merged))
    
    # Group by user (or you could use 'user_id' if that’s the unique identifier)
    merged_df = df.groupby("user").agg({
        "id": "first",          # You can choose how to handle these fields
        "date": lambda x: list(x),      # Gather all dates (if needed)
        "to": merge_to_lists,   # Merge the lists from all rows
        "user_id": "first",      # Assuming user_id is unique per user
        "key_date": lambda x: list(x),  # Gather all key_date values
    }).reset_index()
    
    return merged_df

def email_communication_relationship(df, emails, users):
    # merge all user-day entries to only have one entry per user for the entire timeframe
    email = merge_all_dates_for_each_user(emails)
    email.drop(['date', 'key_date'], axis=1, inplace=True)

    # Precompute a mapping from user email to user index from the users DataFrame.
    user_map = dict(zip(users['user_id'], users.index))
    user_email_map = dict(zip(users['email'], users.index))
    user_node_map = df.groupby('user').apply(lambda group: group.index.tolist()).to_dict()
    
    connections = [
        [], # src
        []  # dst
    ]

    for _, row in tqdm(email.iterrows(), total=email.shape[0]):
        user = user_map[row['user']]
        if user not in user_node_map:
            continue
        for to in row['to']:
            if to not in user_email_map:
                continue
            src_list = user_node_map[user]
            if user_email_map[to] not in user_node_map:
                continue
            dst_list = user_node_map[user_email_map[to]]
            for src, dst in list(itertools.product(src_list, dst_list)):
                connections[0].append(src)
                connections[1].append(dst)
                connections[0].append(dst)
                connections[1].append(src)

    return connections

def email_day_communication_relationship(df, emails, users):
    # Precompute a mapping from merge_key to index in the day DataFrame.
    day_map = dict(zip(df['merge_key'], df.index))
    
    # Precompute a mapping from user email to user index from the users DataFrame.
    user_map = dict(zip(users['email'], users.index))

    connections = [
        [], # src
        []  # dst
    ]

    for _, row in tqdm(emails.iterrows(), desc="Processing email data", total=emails.shape[0]):
        key = row['merge_key']  # Sender's merge_key, e.g. "209_2010-01-02"
        date = row['key_date']  # Date of the email
        
        # Get sender's node from day_map.
        if key not in day_map:
            continue
        sender_node = day_map[key]
        
        # Process each recipient email in the 'to' field.
        for user_email in row['to']:
            # Look up the recipient's user index.
            if user_email not in user_map:
                continue
            recipient_user_index = user_map[user_email]
            
            # Construct the recipient's merge_key (format: "{user_index}_{date}").
            recipient_key = f"{recipient_user_index}_{date}"
            
            # If the recipient's merge_key exists, add an edge.
            if recipient_key in day_map:
                recipient_node = day_map[recipient_key]
                connections[0].append(sender_node)
                connections[1].append(recipient_node)

    return connections

def merge_pc_entries(df):
    merged_df = df.groupby('pc').agg({
        'key_date': lambda x: list(x.unique()),  # unique dates for each PC
        'user_id': lambda x: list(set(sum(x, [])))  # flatten lists and get unique user_ids
    }).reset_index()
    return merged_df

def device_sharing_relationship(df, pcs):
    pc = pcs.explode('pc').groupby(['key_date', 'pc'])['user_id'].apply(list).reset_index()
    pc = merge_pc_entries(pc)
    pc.drop(['key_date'], axis=1, inplace=True)

    user_node_map = df.groupby('user').apply(lambda group: group.index.tolist()).to_dict()

    connections = [
        [], #src
        [] #dst
    ]

    for _, row in pc.iterrows():
        shared_users = row['user_id']
        for user1, user2 in itertools.combinations(shared_users, 2):
            if user1 not in user_node_map or user2 not in user_node_map:
                continue
            src_list = user_node_map[user1]
            dst_list = user_node_map[user2]
            for src, dst in list(itertools.product(src_list, dst_list)):
                connections[0].append(src)
                connections[1].append(dst)
                connections[0].append(dst)
                connections[1].append(src)

    return connections

def device_day_sharing_relationship(df, pcs):
    user_day_index = df.reset_index().set_index(['user', 'key_date'])['index'].to_dict()
    shared_pc_groups = pcs.explode('pc').groupby(['key_date', 'pc'])['user_id'].apply(list).reset_index()

    connections = [[], []]

    for _, row in shared_pc_groups.iterrows():
        shared_users = row['user_id']
        key_date = row['key_date']

        # Create all unique pairs of users who used the same PC
        for user1, user2 in itertools.combinations(shared_users, 2):
            # Get the index positions from feature_user_day
            if (user1, key_date) in user_day_index and (user2, key_date) in user_day_index:
                src = user_day_index[(user1, key_date)]
                dst = user_day_index[(user2, key_date)]

                # Append to connections list
                connections[0].append(src)
                connections[1].append(dst)
    
    return connections

def load_data_cert(path='./rs_data', version='r4.2', conn_type='user'):
    # --- Load and Process User-Date Data ---

    # with open("/fred/oz382/dataset/CERT/r4.2/ExtractedData-comb/user_date.pkl", "rb") as f:
    #     user_date = pickle.load(f)
    # # Map user_index in user_date to a new user_id index (based on users DataFrame)
    # user_date['user_id'] = user_date['user_index'].apply(
    #     lambda x: users[users['user_id'] == x].index[0]
    # )

    # user_date['key_date'] = pd.to_datetime(user_date['date']).dt.date.astype(str)
    users = pd.read_csv(f"{path}/users.csv")

    # --- Process Day Data ---
    day = pd.read_csv(f"{path}/day{version}.csv")
    day['key_date'] = day['starttime'].apply(lambda ts: str(datetime.fromtimestamp(ts).date()))
    day['insider'] = day['insider'].apply(lambda x: 1 if x != 0 else 0)
    
    # --- Process Email Data ---
    email = pd.read_csv(f"{path}/email.csv")
    email.drop(['pc', 'from', 'size', 'attachments', 'content'], axis=1, inplace=True)
    email['to'] = email['to'].str.split(';').apply(
        lambda x: [] if not isinstance(x, list) else [i for i in x if '@dtaa.com' in i]
    )
    email['cc'] = email['cc'].str.split(';').apply(
        lambda x: [] if not isinstance(x, list) else [i for i in x if '@dtaa.com' in i]
    )
    email['bcc'] = email['bcc'].str.split(';').apply(
        lambda x: [] if not isinstance(x, list) else [i for i in x if '@dtaa.com' in i]
    )
    email['to'] = email['to'] + email['cc'] + email['bcc']
    email.drop(['cc', 'bcc'], axis=1, inplace=True)
    email = email[email['to'].map(len) > 0]
    email['user_id'] = email['user'].apply(
        lambda x: users[users['user_id'] == x].index[0]
    )
    email['key_date'] = pd.to_datetime(email['date']).dt.date.astype(str)
    
    # --- Process Logon Data for Device Connections ---
    logon = pd.read_csv(f"{path}/logon.csv")
    logon['key_date'] = pd.to_datetime(logon['date']).dt.date.astype(str)
    pcs = logon.groupby(['user', 'key_date'])['pc'].apply(lambda x: list(set(x))).reset_index()
    pcs['user_id'] = pcs['user'].map(
        lambda x: users[users['user_id'] == x].index[0]
    )

    # filter to contain only key_date that starts from 2010-08
    day = day[day['key_date'].str.startswith('2010-08')].reset_index(drop=True)
    email = email[email['key_date'].str.startswith('2010-08')]
    pcs = pcs[pcs['key_date'].str.startswith('2010-08')]
    
    # --- Build Connection Graph ---
    # Depending on the connection type, build the edge list using a helper function.
    if conn_type == 'none':
        conn = none_homogeneous_relationship(day)
    elif conn_type == 'email':
        conn = email_communication_relationship(day, email, users)
    elif conn_type == 'user':
        conn = user_hierarchical_relationship(day, users)
    elif conn_type == 'device':
        conn = device_sharing_relationship(day, pcs)
    else:
        raise ValueError("Unknown connection type: choose 'email', 'user', or 'device'.")

    conn = np.array(conn)

    num_nodes = day.shape[0]
    # Assume conn is a NumPy array of shape (2, num_edges)
    src = conn[0]
    dst = conn[1]

    # For an undirected graph, include both directions.
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(len(rows), dtype=np.float32)

    adj_sp = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)

    # Convert the SciPy sparse matrix to a torch sparse tensor.
    adj = sparse_mx_to_torch_sparse_tensor(adj_sp)

    # --- Prepare Feature Matrix and Labels ---
    # Drop columns not used as features.
    # filtered_df = day.drop(columns=['key_date', 'merge_key'])
    filtered_df = day.drop(columns=['key_date'])
    features_np = filtered_df.drop(
        ['role', 'b_unit', 'f_unit', 'dept', 'team', 'ITAdmin', 'O', 'C', 'E', 'A', 'N', 'insider'],
        axis=1
    ).to_numpy().astype(float)
    features = torch.from_numpy(features_np).float()

    labels_np = filtered_df['insider'].to_numpy()
    labels = torch.from_numpy(labels_np).long()

    # Optionally standardize and apply PCA to reduce the feature dimensionality.
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.numpy())
    # pca = PCA(n_components=15)
    # principal_components = pca.fit_transform(features_scaled)
    principal_components = features_scaled
    features = torch.FloatTensor(principal_components)
        
    return adj, features, labels

def refine_label_order(labels):
    max_label = labels.max()
    j = 0

    for i in range(labels.max(),0,-1):
        if sum(labels==i) >= IMBALANCE_THRESH and i>j:
            while sum(labels==j) >= IMBALANCE_THRESH and i>j:
                j = j+1
            if i > j:
                head_ind = labels == j
                tail_ind = labels == i
                labels[head_ind] = i
                labels[tail_ind] = j
                j = j+1
            else:
                break
        elif i <= j:
            break

    return labels

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def norm_sparse(adj):#normalize a torch dense tensor for GCN, and change it into sparse.
    adj = adj + torch.eye(adj.shape[0]).to(adj)
    rowsum = torch.sum(adj,1)
    r_inv = 1/rowsum
    r_inv[torch.isinf(r_inv)] = 0.
    new_adj = torch.mul(r_inv.reshape(-1,1), adj)

    indices = torch.nonzero(new_adj).t()
    values = new_adj[indices[0], indices[1]] # modify this based on dimensionality

    return torch.sparse.FloatTensor(indices, values, new_adj.size())

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def find_shown_index(adj, center_ind, steps = 2):
    seen_nodes = {}
    shown_index = []

    if isinstance(center_ind, int):
        center_ind = [center_ind]

    for center in center_ind:
        shown_index.append(center)
        if center not in seen_nodes:
            seen_nodes[center] = 1

    start_point = center_ind
    for step in range(steps):
        new_start_point = []
        candid_point = set(adj[start_point,:].reshape(-1, adj.shape[1]).nonzero()[:,1])
        for i, c_p in enumerate(candid_point):
            if c_p.item() in seen_nodes:
                pass
            else:
                seen_nodes[c_p.item()] = 1
                shown_index.append(c_p.item())
                new_start_point.append(c_p)
        start_point = new_start_point

    return shown_index

