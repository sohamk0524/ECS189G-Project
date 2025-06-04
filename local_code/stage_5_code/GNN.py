import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: [num_nodes, in_features]
        # adj: normalized adjacency matrix as a sparse tensor
        x = torch.spmm(adj, x)          # sparse matrix multiplication
        x = self.linear(x)
        return x

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn3 = GCNLayer(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn3(x, adj)
        return x


input_dim = features.shape[1]  # e.g. 3703 for Citeseer
hidden_dim = 16
output_dim = len(torch.unique(labels))  # number of classes

model = GCN(input_dim, hidden_dim, output_dim)
#model = model.to('cuda')