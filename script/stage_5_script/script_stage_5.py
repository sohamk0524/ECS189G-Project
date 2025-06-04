import torch
import torch.nn as nn
import torch.nn.functional as F
from local_code.stage_5_code import Dataset_Loader_Node_Classification

################ Getting Data from zip #######################

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

# Load the node data
node_data = pd.read_csv("/content/project/ECS189G_Winter_2025_Source_Code_Template/data/stage_5_data/cora/node", sep='\t', header=None)
features = node_data.iloc[:, 1:-1].values  # Shape: (3312, 3703)
labels_raw = node_data.iloc[:, -1].values
node_ids = node_data.iloc[:, 0].values
print(f"node_ids: {node_ids[0:10]}")

# Encode labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels_raw)

# Map node IDs to continuous integer indices
#node_id_map = {node_id: i for i, node_id in enumerate(node_ids)}

# Load edges and map node IDs
edge_data = pd.read_csv("/content/project/ECS189G_Winter_2025_Source_Code_Template/data/stage_5_data/cora/link", sep='\t', header=None)
edges = edge_data.values
edges = edges[(edges[:, 0] != -1) & (edges[:, 1] != -1)]  # Remove invalid edges

# Build the directed graph (B -> A means A points to B)
G = nx.DiGraph()
G.add_nodes_from(range(len(node_ids)))
G.add_edges_from([(dst, src) for src, dst in edges])



##################### Spliting the Data ###################
import numpy as np
from collections import defaultdict

def stratified_sample(labels, n_per_class, random_state=42):
    label_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        label_to_indices[label].append(i)

    np.random.seed(random_state)
    train_idx = []
    for label, indices in label_to_indices.items():
        sampled = np.random.choice(indices, n_per_class, replace=False)
        train_idx.extend(sampled)

    return train_idx

# ✔ Stratified Training Sample: 120 nodes (20 per class for 6 classes)
train_idx = stratified_sample(labels, 20)

# ✔ Remaining indices after training
remaining_indices = np.setdiff1d(np.arange(len(labels)), train_idx)
remaining_labels = labels[remaining_indices]

# ✔ Stratified Testing Sample: 1200 nodes (200 per class from remaining pool)
test_idx_partial = stratified_sample(remaining_labels, 150)
test_idx = remaining_indices[test_idx_partial]  # Correct index mapping

# ✔ Convert to NumPy arrays
train_idx = np.array(train_idx)
test_idx = np.array(test_idx)


## Loading Data
loader = Dataset_Loader()
loader.dataset_name = 'cora'
loader.dataset_source_folder_path = '/content/project/ECS189G_Winter_2025_Source_Code_Template/data/stage_5_data/cora'

loader.custom_train_idx = torch.LongTensor(train_idx)  # your training indices
loader.custom_test_idx = torch.LongTensor(test_idx)    # your testing indices

data = loader.load()
graph = data['graph']
splits = data['train_test_val']

# Extract the elements
features = graph['X']            # torch.FloatTensor, node features
labels = graph['y']              # torch.LongTensor, node labels (encoded)
adj = graph['utility']['A']      # normalized adjacency as torch sparse tensor
idx_train = splits['idx_train']  # train indices as torch.LongTensor
idx_test = splits['idx_test']    # test indices as torch.LongTensor
idx_val = splits['idx_val']      # val indices as torch.LongTensor



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



############ Training ###########


import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
train_loss = []
test_loss = []
train_acc = []
test_acc = []

def train():
    model.train()
    optimizer.zero_grad()
    out = model(features, adj)
    loss = criterion(out[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(idx):
    model.eval()
    with torch.no_grad():
        out = model(features, adj)
        loss = criterion(out[idx], labels[idx])
        logits = out[idx]
        preds = logits.argmax(dim=1)
        true = labels[idx]

        acc = (preds == true).float().mean().item()
        precision = precision_score(true.cpu(), preds.cpu(), average='weighted', zero_division=0)
        recall = recall_score(true.cpu(), preds.cpu(), average='weighted', zero_division=0)
        f1 = f1_score(true.cpu(), preds.cpu(), average='weighted', zero_division=0)

    return acc, precision, recall, f1, loss.item()

# Training loop
for epoch in range(1, 201):  # 200 epochs
    loss = train()
    train_acc_val, _, _, _, _ = evaluate(idx_train)
    val_acc_val, val_prec, val_recall, val_f1, val_loss_val = evaluate(idx_val)

    train_loss.append(loss)
    test_loss.append(val_loss_val)
    train_acc.append(train_acc_val)
    test_acc.append(val_acc_val)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc_val:.4f} | "
              f"Val Acc: {val_acc_val:.4f} | Precision: {val_prec:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")

# Final test accuracy
test_acc, test_prec, test_recall, test_f1, test_loss_val = evaluate(idx_test)
print("\nFinal Test Performance:")
print(f"Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}")
