'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp


class Dataset_Loader(dataset):
    data = None
    dataset_name = None

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(Dataset_Loader, self).__init__(dName, dDescription)

    def adj_normalize(self, mx):
        """normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        onehot_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return onehot_labels

    def load(self):
        """Load citation network dataset"""
        print('Loading {} dataset...'.format(self.dataset_name))

        # load node data from file
        idx_features_labels = np.genfromtxt("{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        onehot_labels = self.encode_onehot(idx_features_labels[:, -1])

        # load link data from file and build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        reverse_idx_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        # convert to pytorch tensors
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(onehot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        # the following part, you can either put them into the setting class or you can leave them in the dataset loader
        # the following train, test, val index are just examples, sample the train, test according to project requirements
        if self.dataset_name == 'cora':
            idx_train = range(140)
            idx_test = range(200, 1200)
            idx_val = range(1200, 1500)
        elif self.dataset_name == 'citeseer':
            idx_train = range(120)
            idx_test = range(200, 1200)
            idx_val = range(1200, 1500)
        elif self.dataset_name == 'pubmed':
            idx_train = range(60)
            idx_test = range(6300, 7300)
            idx_val = range(6000, 6300)
        #---- cora-small is a toy dataset I hand crafted for debugging purposes ---
        elif self.dataset_name == 'cora-small':
            idx_train = range(5)
            idx_val = range(5, 10)
            idx_test = range(5, 10)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        # get the training nodes/testing nodes
        # train_x = features[idx_train]
        # val_x = features[idx_val]
        # test_x = features[idx_test]
        # print(train_x, val_x, test_x)

        train_test_val = {'idx_train': idx_train, 'idx_test': idx_test, 'idx_val': idx_val}
        graph = {'node': idx_map, 'edge': edges, 'X': features, 'y': labels, 'utility': {'A': adj, 'reverse_idx': reverse_idx_map}}
        return {'graph': graph, 'train_test_val': train_test_val}
    

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

