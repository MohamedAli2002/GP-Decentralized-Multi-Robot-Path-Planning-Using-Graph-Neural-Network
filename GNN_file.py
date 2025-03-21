import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Communication_GNN():
    def __init__(self,encoded, adj_mat, num_of_agents, gnn_model=None):
        self.encoded = encoded # ->dict
        self.adj_mat = adj_mat # ->dict
        self.num_of_agents = num_of_agents
        if gnn_model is None:
            seed = 42
            torch.manual_seed(seed)
            self.gnn = GNN(input_dim=128, hidden_dim=128, output_dim=128, num_layers=2).to(device)
        else:
            self.gnn = gnn_model.to(device)

    def begin(self):
        features = {}
        for c in range(len(self.encoded.keys())): # for every case
            features_steps = {}
            for s in range(len(self.encoded[c])): # for every step
                encoded_step = torch.tensor(self.encoded[c][s], dtype=torch.float32).to(device)
                adj = torch.tensor(self.adj_mat[c][s], dtype=torch.float32).to(device)
                edge_index, _ = dense_to_sparse(adj)
                gnn_features = self.gnn(encoded_step,edge_index)
                features_steps[s] = gnn_features
            features[c] = features_steps
        return features
                

class GNN(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=128, num_layers = 2):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        self.convs.append(GCNConv(hidden_dim, output_dim))
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
        return x
