import dgl
import dgl.function as fn
from dgl.data import DGLDataset
import torch
import CommonModules as CM
import os
import collections
from sklearn.model_selection import train_test_split
import scipy.io

from datasets.base.utils import get_line_graph
import numpy as np
import pandas as pd
from icecream import ic


class SetsLineGraph(DGLDataset):
    def __init__(self, name, subset=None, no_node_features=False, homo=True, aggregate=False,
                 bidirected=True, initialized_instance=None):
        self.homo = homo
        self.aggregate = aggregate
        self.bidirected = bidirected
        if initialized_instance is None:
            super().__init__(name=name)
        else:
            self = initialized_instance
        graph = self.graphs[0]
        if subset == "train":
            self.nodes_mask = graph.ndata["train_mask"]
        elif subset == "val":
            self.nodes_mask = graph.ndata["val_mask"]
        elif subset == "test":
            self.nodes_mask = graph.ndata["test_mask"]
        else:
            self.nodes_mask = None
        if no_node_features:  # Exclude any node features
            self.nfeats = torch.eye(graph.num_nodes())
        else:
            self.nfeats = graph.ndata["nfeat"]
            ic(self.nfeats.shape)
    #         self.efeats = self.graph.edata["efeat"]

    def process(self):
        df_train = self.df_train
        df_val = self.df_val
        df_test = self.df_test
        train_embeds, train_labels = self.train_embeds, self.train_labels
        val_embeds, val_labels = self.val_embeds, self.val_labels
        test_embeds, test_labels = self.test_embeds, self.test_labels

        train_embeds = train_embeds.astype(np.float32)
        val_embeds = val_embeds.astype(np.float32)
        test_embeds = test_embeds.astype(np.float32)
        train_labels[train_labels > 0] = 1
        val_labels[val_labels > 0] = 1
        test_labels[test_labels > 0] = 1
        #         print(type(train_embeds))

        self.df = df = pd.concat([df_train, df_val, df_test], axis=0)
        train_embeds = torch.from_numpy(train_embeds)
        val_embeds = torch.from_numpy(val_embeds)
        test_embeds = torch.from_numpy(test_embeds)
        train_labels = torch.from_numpy(train_labels)
        val_labels = torch.from_numpy(val_labels)
        test_labels = torch.from_numpy(test_labels)
        edge_features = torch.cat([train_embeds, val_embeds, test_embeds])
        edge_labels = torch.cat([train_labels, val_labels, test_labels])
        edge_labels = edge_labels.type(torch.long)
        #         nodes_data = pd.read_csv('./members.csv')
        #         node_features = torch.from_numpy(nodes_data['Age'].to_numpy())
        #         node_labels = torch.from_numpy(nodes_data['Club'].astype('category').cat.codes.to_numpy())
        #         edge_features = torch.from_numpy(edges_data['Weight'].to_numpy())

        # Connect edge nodes if they are parallel (reviewers write multiple reviews for the same product)
        edges_src = torch.from_numpy(df['SENDER_id'].to_numpy()).clone()
        edges_dst = torch.from_numpy(df['RECEIVER_id'].to_numpy()).clone()
        seen_dict = collections.defaultdict(int)
        parallel_edge_group = {}  # {(src, dst): [edge indices]}
        # parallel_edge_group_id = {}  # {(src, dst): edge group id}
        edges_src_list = edges_src.tolist()
        edges_dst_list = edges_dst.tolist()
        for index, src, dst in zip(range(len(edges_src)), edges_src_list, edges_dst_list):
            if (src, dst) not in parallel_edge_group:
                parallel_edge_group[(src, dst)] = [index]
                # parallel_edge_group_id[(src, dst)] = len(parallel_edge_group)
            else:
                parallel_edge_group[(src, dst)].append(index)
        max_value = max(max(edges_src), max(edges_dst)) + 1
        ic(len(parallel_edge_group))
        for edge_group, edge_indices in parallel_edge_group.items():
            src, dst = edge_group
            if seen_dict[src] > 0 or seen_dict[dst] > 0:
                for edge_index in edge_indices:
                    edges_src[edge_index] = max_value
                    edges_dst[edge_index] = max_value + 1
                max_value += 2
                seen_dict[src] += 1
                seen_dict[dst] += 1
            else:
                seen_dict[src] += 1
                seen_dict[dst] += 1
        graph_1 = dgl.graph((edges_src, edges_dst))
        # graph_2 = dgl.graph((edges_src, edges_dst))
        # graph_3 = dgl.graph((edges_src, edges_dst))

        # Connect edge nodes if they share the same src node (same reviewer)
        # Will make dst nodes as all different nodes so no connection between original dst nodes
        edges_src = torch.from_numpy(df['SENDER_id'].to_numpy()).clone()
        edges_dst = torch.from_numpy(df['RECEIVER_id'].to_numpy()).clone()
        seen_dict = {}
        parallel_edge_dict = {}
        edges_src_list = edges_src.tolist()
        edges_dst_list = edges_dst.tolist()
        max_value = max(max(edges_src), max(edges_dst)) + 1
        # unique_values = edges_dst.unique().shape[0]

        # Make sure the dst node only appear once in the new graph to construct line graph
        # and edges with the same src node will be automatically connected with each other
        for index in range(edges_dst.shape[0]):
            seen_dict[edges_dst_list[index]] = 0
        # Ignore edges already appeared in S1
        for index in range(edges_dst.shape[0]):
            if (edges_src_list[index], edges_dst_list[index]) in parallel_edge_dict:
                continue
            else:
                parallel_edge_dict[(edges_src_list[index], edges_dst_list[index])] = 1
            value = edges_dst_list[index]
            if seen_dict[value] > 0:
                edges_dst[index] = max_value
                max_value += 1
                # seen_dict[value] += 1
            else:
                seen_dict[value] += 1
        # graph_1 = dgl.graph((edges_src, edges_dst))
        graph_2 = dgl.graph((edges_src, edges_dst))
        # graph_3 = dgl.graph((edges_src, edges_dst))

        # Connect edge nodes if they share the same dst node (same product)
        # Will make src nodes as all different nodes so no connection between original src nodes
        edges_src = torch.from_numpy(df['SENDER_id'].to_numpy()).clone()
        edges_dst = torch.from_numpy(df['RECEIVER_id'].to_numpy()).clone()
        seen_dict = {}
        parallel_edge_dict = {}
        edges_src_list = edges_src.tolist()
        edges_dst_list = edges_dst.tolist()
        max_value = max(max(edges_src), max(edges_dst)) + 1
        # unique_values = edges_dst.unique().shape[0]

        for index in range(edges_src.shape[0]):
            seen_dict[edges_src_list[index]] = 0
        for index in range(edges_src.shape[0]):
            if (edges_src_list[index], edges_dst_list[index]) in parallel_edge_dict:
                continue
            else:
                parallel_edge_dict[(edges_src_list[index], edges_dst_list[index])] = 1
            value = edges_src_list[index]
            if seen_dict[value] > 0:
                edges_src[index] = max_value
                max_value += 1
                # seen_dict[value] += 1
            else:
                seen_dict[value] += 1
        graph_3 = dgl.graph((edges_src, edges_dst))

        self.graphs = [graph_1, graph_2, graph_3]

        n_train = train_embeds.shape[0]
        n_val = val_embeds.shape[0]
        n_test = test_embeds.shape[0]
        n_nodes = graph_1.number_of_edges()
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        edge_start_index = 0
        train_mask[edge_start_index:edge_start_index + n_train] = True
        val_mask[edge_start_index + n_train:edge_start_index + n_train + n_val] = True
        test_mask[edge_start_index + n_train + n_val:edge_start_index + n_train + n_val + n_test] = True
        # After converting into line graph, edge features will be node features
        for index, graph in enumerate(self.graphs):
            graph.edata["eid"] = torch.arange(edge_features.shape[0], dtype=torch.int)
            graph.edata["nfeat"] = edge_features
            graph.edata["train_mask"] = train_mask
            graph.edata["val_mask"] = val_mask
            graph.edata["test_mask"] = test_mask
            graph.edata["label"] = edge_labels
            # G = dgl.graph(([0,1,2],[1,2,1]))
            # get_line_graph(G, False)
            if index > 0:  # For S2 and S3 edges
                remove_parallel = True
            else:  # For S1 edges
                remove_parallel = False
            graph = get_line_graph(graph, as_directed=False, copy=False, remove_parallel=remove_parallel)
            l = graph.ndata["eid"]
            assert all(l[i] <= l[i + 1] for i in range(len(l) - 1))  # Make sure eids are sorted so they are aligned
            self.graphs[index] = graph.add_self_loop()
        # ic(sum(self.graphs[0].ndata["label"]==self.graphs[1].ndata["label"]),
        #    sum(self.graphs[0].ndata["label"]==self.graphs[2].ndata["label"]),
        #    sum(self.graphs[1].ndata["label"]==self.graphs[2].ndata["label"]))

    def __getitem__(self, i):
        return {
            "g": self.graphs,
            #             "efeats": self.efeats,
            "nfeats": self.nfeats,
            "labels": self.graphs[0].ndata["label"],
            "nodes_mask": self.nodes_mask,
            # "edges_id": self.edges_id,
            #             "edges_src": self.edges_src,
            #             "edges_dst": self.edges_dst
        }

    def __len__(self):
        return 1

# if __name__ == "__main__":
#     dataset = Yelpchi3Sets()
#     graph = dataset[0]
#     # graph = dgl.to_homogeneous(graph, ["user", "transaction"])
#     print(graph)
