import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionMLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_actions=5):
        super(ActionMLP, self).__init__()
        seed = 42
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        seed = 42
        torch.manual_seed(seed)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        seed = 42
        torch.manual_seed(seed)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        seed = 42
        torch.manual_seed(seed)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        seed = 42
        torch.manual_seed(seed)
        self.fc3 = nn.Linear(hidden_dim, num_actions)
        seed = 42
        torch.manual_seed(seed)
        self.dropout = nn.Dropout(0.2)  # Dropout for regularization

    def forward(self, x):
        seed = 42
        torch.manual_seed(seed)
        x = F.relu(self.bn1(self.fc1(x)))
        seed = 42
        torch.manual_seed(seed)
        x = self.dropout(x)
        seed = 42
        torch.manual_seed(seed)
        x = F.relu(self.bn2(self.fc2(x)))
        seed = 42
        torch.manual_seed(seed)
        x = self.dropout(x)
        seed = 42
        torch.manual_seed(seed)
        x = self.fc3(x)
        seed = 42
        torch.manual_seed(seed)
        return F.softmax(x, dim=-1)