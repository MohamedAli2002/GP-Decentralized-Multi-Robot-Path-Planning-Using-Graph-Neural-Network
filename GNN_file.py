import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Communication_GNN:
    def __init__(self, encoded, adj_mat, num_of_agents, gnn_model=None):
        self.encoded = encoded  # dict
        self.adj_mat = adj_mat  # dict
        self.num_of_agents = num_of_agents
        if gnn_model is None:
            seed = 42
            torch.manual_seed(seed)
            self.gnn = PaperGNN(input_dim=128, output_dim=128, K=2).to(device)
        else:
            self.gnn = gnn_model.to(device)

    def begin(self):
        features = {}
        for c in self.encoded.keys():  # Iterate directly over keys
            features_steps = {}
            for s in range(len(self.encoded[c])):
                # Move tensors to device
                encoded_step = torch.tensor(self.encoded[c][s], dtype=torch.float32).to(device)
                adj = torch.tensor(self.adj_mat[c][s], dtype=torch.float32).to(device)
                # Forward pass
                gnn_features = self.gnn(encoded_step, adj)
                # gnn_features = self.gnn(encoded_step, St)
                features_steps[s] = gnn_features
            features[c] = features_steps
        return features

class PaperGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, K=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.K = K
        
        self.A = nn.ParameterList([
            nn.Parameter(torch.Tensor(input_dim, output_dim))
            for _ in range(K)
        ])
        
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        # seed = 42
        # torch.manual_seed(seed)
        self.A = nn.parameter.Parameter(torch.Tensor(K,input_dim,output_dim))
        self.bias = nn.parameter.Parameter(torch.Tensor(output_dim))
        # self.A = nn.ParameterList([
        #     nn.Parameter(torch.Tensor(input_dim, output_dim))
        #     for _ in range(K)
        # ])

        # self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        for a in self.A:
            nn.init.xavier_uniform_(a)
        nn.init.zeros_(self.bias)
        # seed = 42
        # torch.manual_seed(seed)
        stdv = 1. / math.sqrt(self.input_dim * self.K)
        self.A.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        # for a in self.A:
        #     nn.init.xavier_uniform_(a)
        # nn.init.zeros_(self.bias)

    def forward(self, X, St):
        out = 0
        X_k = X 
        
        X_k = X

        for k in range(self.K):
            # Add term for current shift power
            out = out + torch.mm(X_k, self.A[k])
            
            if k < self.K - 1:  # No need to compute beyond K-1
                X_k = torch.sparse.mm(St, X_k)
        
                X_k = St @ X_k
                # X_k = torch.sparse.mm(St, X_k)

        return out + self.bias

class PaperGNN(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=128, 
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=128,
                 num_layers=2, K=3):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        

        # First layer
        self.convs.append(PaperGraphConv(input_dim, hidden_dim, K))
        

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(PaperGraphConv(hidden_dim, hidden_dim, K))
        

        # Final layer
        self.convs.append(PaperGraphConv(hidden_dim, output_dim, K))

    def forward(self, x, St):
        for i, conv in enumerate(self.convs):
            x = conv(x, St)
            if i != self.num_layers - 1:  # ReLU for all but last layer
                x = F.relu(x)
        return x
