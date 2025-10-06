import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, HeteroConv, HGTConv
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.nn.models import Node2Vec
# from utils.heterogeneous_dataloader import get_edges_dict
import time

class ResHybNet(nn.Module):
    def __init__(self,input_dim, output_dim, cnn='' ,gnn='',residual='' ):
        super(ResHybNet,self).__init__()
        
        if cnn=='':
            self.mode= 'gnn'
        elif gnn=='':
            self.mode= 'cnn'
        else:
            self.mode= 'hybrid'
        #initialize cnn layer base on model type
        if cnn=='CNN':
            self.conv1 = nn.Sequential(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),                              
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)  
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),                               
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2) 
            )
        # initialize gnn mode based on model type:
        if gnn== 'GCN':
            self.gnn1= GCNConv(input_dim, 16)
            self.gnn2= GCNConv(16, output_dim)
        elif gnn=='GAT':
            self.gnn1= GATConv(input_dim, 16)
            self.gnn2= GATConv(16, output_dim)
        elif gnn=='SAGE':
            self.gnn1= SAGEConv(input_dim, 16)
            self.gnn2= SAGEConv(16, output_dim)
            
        # initialize output fc layer based on feature number:      
        self.output = nn.Linear(32 * (input_dim // 4), 2)
        self.output_g = nn.Linear(input_dim, 2)
            
        if residual=='YES':
            self.res= True
        elif residual == 'NO':
            self.res= False
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        if self.mode== 'cnn':
            x_c = torch.unsqueeze(x,dim=1)
            x_c = self.conv1(x_c)                  
            x_c = self.conv2(x_c)                
            x_c = x_c.view(x_c.size(0),-1)       
            out = self.output(x_c)
            
        elif self.mode== 'gnn':
            x_g = self.gnn1(x, edge_index)
            x_g = F.relu(x_g)
            x_g = F.dropout(x_g, training=self.training)
            x_g = self.gnn2(x_g, edge_index)
            out = self.output_g(x_g)

        else:
            x_g = self.gnn1(x, edge_index)
            x_g = F.relu(x_g)
            x_g = F.dropout(x_g, training=self.training)
            x_g = self.gnn2(x_g, edge_index)
            
            
            # add two channel
            if self.res:
                x_dual= x+x_g
            else:
                x_dual= x_g
            
            x_dual = torch.unsqueeze(x_dual,dim=1)
            x_dual = self.conv1(x_dual)                  
            x_dual = self.conv2(x_dual)                
            x_dual = x_dual.view(x_dual.size(0),-1)
            out= self.output(x_dual)

        return F.log_softmax(out, dim=1)
    
class EnhancedResHybNet(nn.Module):
    def __init__(self, input_dim, output_dim, cnn='CNN', gnn='GCN', residual='YES',
                 use_node2vec=False, use_pos_enc=False, num_pos_enc=8):
        super(EnhancedResHybNet, self).__init__()
        
        self.use_cnn = cnn.upper() == 'CNN'
        self.use_gnn = gnn.upper() != ''
        self.residual = residual.upper() == 'YES'
        self.input_dim = input_dim
        self.use_node2vec = use_node2vec
        self.use_pos_enc = use_pos_enc
        self.num_pos_enc = num_pos_enc

        # Input feature dimension extension if needed
        extra_input_dim = 0
        if use_node2vec:
            self.node2vec = None  # to be initialized in setup_node2vec()
            self.node2vec_dim = 32
            extra_input_dim += self.node2vec_dim
        if use_pos_enc:
            self.pos_fc = nn.Linear(num_pos_enc, num_pos_enc)
            extra_input_dim += num_pos_enc

        self.total_input_dim = input_dim + extra_input_dim

        if self.use_cnn:
            self.cnn_branch = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
        if self.use_gnn:
            if gnn.upper() == 'GCN':
                GNNLayer = GCNConv
            elif gnn.upper() == 'GAT':
                GNNLayer = GATConv
            elif gnn.upper() == 'SAGE':
                GNNLayer = SAGEConv
            else:
                raise ValueError("Unsupported GNN type")

            self.gnn1 = GNNLayer(self.total_input_dim, 32)
            self.bn1 = nn.BatchNorm1d(32)
            self.gnn2 = GNNLayer(32, 32)
            self.bn2 = nn.BatchNorm1d(32)

            if self.residual and self.total_input_dim != 32:
                self.res_fc = nn.Linear(self.total_input_dim, 32)
            else:
                self.res_fc = None

        combined_dim = 0
        if self.use_cnn:
            combined_dim += 32
        if self.use_gnn:
            combined_dim += 32

        self.classifier = nn.Linear(combined_dim, 2)

    def setup_node2vec(self, edge_index, num_nodes, device):
        self.node2vec = Node2Vec(edge_index, embedding_dim=self.node2vec_dim,
                                 walk_length=20, context_size=10, walks_per_node=10,
                                 num_nodes=num_nodes, sparse=True).to(device)
        self.node2vec.reset_parameters()

    def add_positional_encoding(self, data):
        retry = 0
        while retry < 3:
            try:
                transform = AddLaplacianEigenvectorPE(self.num_pos_enc)
                data = transform(data)
                return data
            except Exception as e:
                print(f"Error in computing Laplacian eigenvectors: {e}")
                time.sleep(1)
                retry += 1
        else:
            raise RuntimeError("Failed to compute Laplacian eigenvectors after 3 attempts.")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if self.input_dim is None:
            self.input_dim = x.size(1)

        # Optional: Node2Vec embedding
        if self.use_node2vec and self.node2vec is not None:
            node_ids = torch.arange(x.size(0), device=x.device)
            node2vec_emb = self.node2vec(node_ids)
            x = torch.cat([x, node2vec_emb], dim=1)

        # Optional: Positional encodings from Laplacian eigenvectors
        if self.use_pos_enc and hasattr(data, 'laplacian_eigenvector_pe'):
            pos_enc = self.pos_fc(data.laplacian_eigenvector_pe.to(x.device))
            pos_enc = pos_enc.to(x.device)
            x = torch.cat([x, pos_enc], dim=1)

        features = []

        if self.use_gnn:
            x_g = self.gnn1(x, edge_index)
            x_g = self.bn1(x_g)
            x_g = F.relu(x_g)
            x_g = F.dropout(x_g, p=0.5, training=self.training)
            x_g = self.gnn2(x_g, edge_index)
            x_g = self.bn2(x_g)
            x_g = F.relu(x_g)

            if self.residual:
                x_proj = x if self.res_fc is None else self.res_fc(x)
                x_g = x_g + x_proj

            features.append(x_g)

        if self.use_cnn:
            x_cnn = x.unsqueeze(1)
            x_cnn = self.cnn_branch(x_cnn)
            x_cnn = x_cnn.view(x_cnn.size(0), -1)
            features.append(x_cnn)

        if len(features) > 1:
            x_combined = torch.cat(features, dim=1)
        else:
            x_combined = features[0]

        out = self.classifier(x_combined)
        return F.log_softmax(out, dim=1)
