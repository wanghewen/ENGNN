import dgl
import dgl.function as fn
import scipy.sparse
from dgl.data import DGLDataset
import torch
import CommonModules as CM
import os
import collections
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from icecream import ic


# Refer to https://docs.dgl.ai/en/latest/new-tutorial/6_load_data.html
# And https://github.com/minhduc0711/gcn/blob/2d09dd2f5c5400e4b53f74e2b35a5a612904ae48/src/data/datasets.py
class EdgeAttributed(DGLDataset):
    def __init__(self, name, data_folder, subset=None, no_node_features=False,
                 initialized_instance=None, add_self_loop=True,
                 randomize_train_test=False, randomize_by_node=False, num_classes=None,
                 sample_way=None, sample_size=None, reduced_dim=None, use_linegraph=False, as_directed=True,
                 as_dynamic=False, test_portion_index=9):
        # This will introduce zero attributed edges, which will affect training process like batch normalization
        self.add_self_loop = add_self_loop
        self.randomize_train_test = randomize_train_test
        self.randomize_by_node = randomize_by_node
        self.num_classes = num_classes
        self.use_linegraph = use_linegraph
        self.as_directed = as_directed
        self.as_dynamic = as_dynamic  # Split the dataset into 10 portions as 1 train/1 val/8 test
        # When as_dynamic, generate the graph using the test_portion_number
        # E.g., When test_portion_index==2, there will be 1 train/1 val/1 test
        # When test_portion_index==9, there will be 1 train/1 val/8 test
        if self.as_dynamic:
            assert 2 <= test_portion_index <= 9
            assert not self.randomize_train_test
            assert not self.randomize_by_node
        self.test_portion_index = test_portion_index
        if initialized_instance is None:
            if reduced_dim:
                data_folder = os.path.join(data_folder, f"reduce_dim_{reduced_dim}")
            if sample_size:
                data_folder = data_folder + f"_{sample_way}_sampled_size={sample_size}"
            else:
                data_folder = data_folder
            self.df_train = pd.read_csv(os.path.join(data_folder, "train_edges.csv"))
            self.df_val = pd.read_csv(os.path.join(data_folder, "val_edges.csv"))
            self.df_test = pd.read_csv(os.path.join(data_folder, "test_edges.csv"))
            self.train_embeds, self.train_labels = CM.IO.ImportFromPkl(
                os.path.join(data_folder, "train_embeds_labels.pkl"))
            self.val_embeds, self.val_labels = CM.IO.ImportFromPkl(
                os.path.join(data_folder, "val_embeds_labels.pkl"))
            self.test_embeds, self.test_labels = CM.IO.ImportFromPkl(
                os.path.join(data_folder, "test_embeds_labels.pkl"))
            if len(self.train_labels.shape) > 1:
                label_column_mask = np.array(self.train_labels.sum(0) > 10).reshape(-1)  # Exclude class with too few
                # samples
                self.train_labels = self.train_labels[:, label_column_mask]  # Exclude class with too few samples
                self.val_labels = self.val_labels[:, label_column_mask]  # Exclude class with too few samples
                self.test_labels = self.test_labels[:, label_column_mask]  # Exclude class with too few samples
            super().__init__(name=name)
        else:
            self = initialized_instance
        if subset == "train":
            self.edges_mask = self.graph.edata["train_mask"]
        elif subset == "val":
            self.edges_mask = self.graph.edata["val_mask"]
        elif subset == "test":
            self.edges_mask = self.graph.edata["test_mask"]
        else:
            self.edges_mask = None
        if no_node_features:  # Exclude any node features
            self.nfeats = None
        self.nodes_mask = None

    def _concatenate_dataset(self):
        df = pd.concat([self.df_train, self.df_val, self.df_test])
        if isinstance(self.train_embeds, (np.ndarray, torch.Tensor)):
            embeds = np.concatenate([self.train_embeds, self.val_embeds, self.test_embeds], axis=0)
        else:
            embeds = scipy.sparse.vstack([self.train_embeds, self.val_embeds, self.test_embeds])
        if isinstance(self.train_labels, (np.ndarray, torch.Tensor)):
            labels = np.concatenate([self.train_labels, self.val_labels, self.test_labels], axis=0)
        else:
            labels = scipy.sparse.vstack([self.train_labels, self.val_labels, self.test_labels])
        return df, embeds, labels

    def process(self):
        print("Number of edges for original graph:", self.df_train.shape[0] + self.df_val.shape[0] + self.df_test.shape[0])
        print("Number of Nodes for original graph:",
              len(set(self.df_train['SENDER_id']) | set(self.df_train['RECEIVER_id']) |
                 set(self.df_val['SENDER_id']) | set(self.df_val['RECEIVER_id']) |
                  set(self.df_test['SENDER_id']) | set(self.df_test['RECEIVER_id'])))
        if self.randomize_train_test and not self.randomize_by_node:
            print("Randomly split dataset by edges...")
            df, embeds, labels = self._concatenate_dataset()
            if self.num_classes:
                labels = labels[:, :self.num_classes].A
                self.labels = labels
            index = np.arange(df.shape[0])
            train_index, test_index = train_test_split(index, test_size=0.2, shuffle=True, random_state=12)
            train_index, val_index = train_test_split(train_index, test_size=0.2, shuffle=True, random_state=12)
            df_train, df_val, df_test = df.iloc[train_index], df.iloc[val_index], df.iloc[test_index]
            self.train_embeds, self.val_embeds, self.test_embeds = embeds[train_index], embeds[val_index], embeds[test_index]
            self.train_labels, self.val_labels, self.test_labels = labels[train_index], labels[val_index], labels[test_index]
        elif self.randomize_by_node:
            print("Randomly split dataset by nodes...")
            df, embeds, labels = self._concatenate_dataset()
            if self.num_classes:
                labels = labels[:, :self.num_classes].A
                self.labels = labels
            # ###########DEBUG############
            # df = df.drop_duplicates(subset=['SENDER_id', 'RECEIVER_id'], ignore_index=True)
            # ############################
            all_node_list = np.array(list(set(df["SENDER_id"]) | set(df["RECEIVER_id"])))
            index = np.arange(len(all_node_list))
            node_train_index, node_test_index = train_test_split(index, test_size=0.2, shuffle=True, random_state=12)
            node_train_index, node_val_index = train_test_split(node_train_index, test_size=0.2, shuffle=True, random_state=12)
            train_index = (df["SENDER_id"].isin(all_node_list[node_train_index]) | df["RECEIVER_id"].isin(
                all_node_list[node_train_index])).to_numpy()
            val_index = ((df["SENDER_id"].isin(all_node_list[node_val_index]) | df["RECEIVER_id"].isin(
                all_node_list[node_val_index])) & ~train_index).to_numpy()
            test_index = ((df["SENDER_id"].isin(all_node_list[node_test_index]) | df["RECEIVER_id"].isin(
                all_node_list[node_test_index])) & ~train_index).to_numpy()
            df_train, df_val, df_test = df.iloc[train_index], df.iloc[val_index], df.iloc[test_index]
            self.train_embeds, self.val_embeds, self.test_embeds = embeds[train_index], embeds[val_index], embeds[test_index]
            self.train_labels, self.val_labels, self.test_labels = labels[train_index], labels[val_index], labels[test_index]
        elif self.as_dynamic:
            print("Split using dynamic graph logic...")
            df, embeds, labels = self._concatenate_dataset()
            if self.num_classes:
                labels = labels[:, :self.num_classes].A
                self.labels = labels
            index = np.arange(df.shape[0])
            train_index, test_index = train_test_split(index, test_size=0.2, shuffle=False)
            train_index, val_index = train_test_split(train_index, test_size=0.5, shuffle=False)
            if self.test_portion_index >= 9:
                pass
            else:
                test_index, _ = train_test_split(test_index, train_size=(self.test_portion_index-1)/8, shuffle=False)
            df_train, df_val, df_test = df.iloc[train_index], df.iloc[val_index], df.iloc[test_index]
            self.train_embeds, self.val_embeds, self.test_embeds = embeds[train_index], embeds[val_index], embeds[test_index]
            self.train_labels, self.val_labels, self.test_labels = labels[train_index], labels[val_index], labels[test_index]
        else:
            print("Not randomly split dataset...")
            df_train, df_val, df_test = self.df_train, self.df_val, self.df_test
            if self.num_classes:
                self.train_labels = self.train_labels[:, :self.num_classes].A
                self.val_labels = self.val_labels[:, :self.num_classes].A
                self.test_labels = self.test_labels[:, :self.num_classes].A
            self.train_labels, self.val_labels, self.test_labels = self.train_labels, self.val_labels, self.test_labels

        self.train_embeds = self.train_embeds.astype(np.float32)
        self.val_embeds = self.val_embeds.astype(np.float32)
        self.test_embeds = self.test_embeds.astype(np.float32)
        if isinstance(self.train_labels, (np.ndarray, torch.Tensor)) and len(self.train_labels.shape) > 1 and self.train_labels.shape[1] > 1:
            self.train_labels[self.train_labels > 0] = 1
            self.val_labels[self.val_labels > 0] = 1
            self.test_labels[self.test_labels > 0] = 1

            self.train_labels[self.train_labels < 0] = 0
            self.val_labels[self.val_labels < 0] = 0
            self.test_labels[self.test_labels < 0] = 0

        df = pd.concat([df_train, df_val, df_test], axis=0)
        if isinstance(self.train_embeds, (np.ndarray, torch.Tensor)):
            self.train_embeds = torch.from_numpy(self.train_embeds)
            self.val_embeds = torch.from_numpy(self.val_embeds)
            self.test_embeds = torch.from_numpy(self.test_embeds)
            edge_features = torch.cat([self.train_embeds, self.val_embeds, self.test_embeds])
        else:
            edge_features = scipy.sparse.vstack([self.train_embeds, self.val_embeds, self.test_embeds])
        if isinstance(self.train_labels, (np.ndarray, torch.Tensor)):
            self.train_labels = torch.from_numpy(self.train_labels).long()
            self.val_labels = torch.from_numpy(self.val_labels).long()
            self.test_labels = torch.from_numpy(self.test_labels).long()
            edge_labels = torch.cat([self.train_labels, self.val_labels, self.test_labels])
        else:
            edge_labels = scipy.sparse.vstack([self.train_labels, self.val_labels, self.test_labels])

        edges_src = torch.from_numpy(df['SENDER_id'].to_numpy())
        edges_dst = torch.from_numpy(df['RECEIVER_id'].to_numpy())

        self.graph = dgl.graph((edges_src, edges_dst))
        ic(self.graph.number_of_nodes(), edge_features.shape, edges_src.shape, edges_dst.shape, edge_labels.shape)
        #         self.graph.ndata[dgl.NTYPE] = node_types
        self.efeats = edge_features
        if isinstance(edge_features, (np.ndarray, torch.Tensor)):
            self.graph.edata['efeat'] = edge_features
            self.graph.update_all(fn.copy_e('efeat', 'm'), fn.mean('m', 'nfeat'))  # i.e. self.graph.ndata["nfeat"]
            self.nfeats = self.graph.ndata["nfeat"]
        else:
            self.nfeats = self.efeats  # Dummy nfeats! Should never be used!
        if len(edge_labels.shape) == 1:
            edge_labels = edge_labels.reshape(-1, 1)
        if isinstance(edge_labels, (np.ndarray, torch.Tensor)):
            self.graph.edata['label'] = edge_labels
        self.labels = edge_labels
        if self.add_self_loop:
            self.graph = dgl.add_self_loop(self.graph)  # only 1 graph in dataset
            self.labels = self.graph.edata["label"]
            self.efeats = self.graph.edata["efeat"]
        self.edges_id = torch.arange(self.graph.number_of_edges())
        # self.edges_src, self.edges_dst = self.graph.find_edges(self.edges_id)

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        #         m_edges = self.graph.number_of_edges()
        n_edges = self.graph.number_of_edges()
        n_train = self.train_embeds.shape[0]
        n_val = self.val_embeds.shape[0]
        n_test = self.test_embeds.shape[0]
        train_mask = torch.zeros(n_edges, dtype=torch.bool)
        val_mask = torch.zeros(n_edges, dtype=torch.bool)
        test_mask = torch.zeros(n_edges, dtype=torch.bool)
        edge_start_index = 0
        train_mask[edge_start_index:edge_start_index + n_train] = True
        val_mask[edge_start_index + n_train:edge_start_index + n_train + n_val] = True
        test_mask[edge_start_index + n_train + n_val:edge_start_index + n_train + n_val + n_test] = True
        self.graph.edata['train_mask'] = train_mask
        self.graph.edata['val_mask'] = val_mask
        self.graph.edata['test_mask'] = test_mask

    def __getitem__(self, i):
        return {
            "g": self.graph,
            "efeats": self.efeats,
            "nfeats": self.nfeats,  # Dummy nfeats! Should never be used!
            "labels": self.labels,
            "edges_mask": self.edges_mask,
            "edges_id": self.edges_id,
            "nodes_mask": self.nodes_mask,
            # "edges_src": self.edges_src,
            # "edges_dst": self.edges_dst
        }

    def __len__(self):
        return 1
