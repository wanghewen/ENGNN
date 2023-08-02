import dgl
from dgl.data import DGLDataset
import torch
import collections
import numpy as np
import pandas as pd

from datasets.base.utils import get_line_graph, get_ppr_graph
from icecream import ic


# Refer to https://docs.dgl.ai/en/latest/new-tutorial/6_load_data.html
# And https://github.com/minhduc0711/gcn/blob/2d09dd2f5c5400e4b53f74e2b35a5a612904ae48/src/data/datasets.py
class Sets(DGLDataset):
    # 24141 nodes, 141141 edges ->
    def __init__(self, name, edge_to_node_data_class=None,
                 edge_to_node_data_folder=None, subset=None, no_node_features=False,
                 homo=True, aggregate=False, bidirected=True,
                 initialized_instance=None, add_self_loop=True, topppr_k=1000, alpha=None):
        self.homo = homo
        self.aggregate = aggregate
        self.bidirected = bidirected
        self.edge_to_node_data_class = edge_to_node_data_class
        self.edge_to_node_data_folder = edge_to_node_data_folder
        self.add_self_loop = add_self_loop
        self.topppr_k = topppr_k
        self.alpha = alpha
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
        ic(edge_labels.sum(0), edge_labels.shape[0], "unbalance ratio:", edge_labels.sum(0) / edge_labels.shape[0])
        #         nodes_data = pd.read_csv('./members.csv')
        #         node_features = torch.from_numpy(nodes_data['Age'].to_numpy())
        #         node_labels = torch.from_numpy(nodes_data['Club'].astype('category').cat.codes.to_numpy())
        #         edge_features = torch.from_numpy(edges_data['Weight'].to_numpy())
        edges_src = torch.from_numpy(df['SENDER_id'].to_numpy()).clone()
        edges_dst = torch.from_numpy(df['RECEIVER_id'].to_numpy()).clone()
        seen_dict = collections.defaultdict(int)
        parallel_edge_group = {}  # {(src, dst): [edge indices]}
        # parallel_edge_group_id = {}  # {(src, dst): edge group id}
        edges_src_list = edges_src.tolist()
        edges_dst_list = edges_dst.tolist()
        # Find all parallel E-Node groups
        for index, src, dst in zip(range(len(edges_src)), edges_src_list, edges_dst_list):
            if (src, dst) not in parallel_edge_group:
                parallel_edge_group[(src, dst)] = [index]
                # parallel_edge_group_id[(src, dst)] = len(parallel_edge_group)
            else:
                parallel_edge_group[(src, dst)].append(index)
        max_value = max(max(edges_src), max(edges_dst)) + 1
        ic(len(parallel_edge_group))
        # Add connections with in parallel E-Nodes
        for edge_group, edge_indices in parallel_edge_group.items():
            src, dst = edge_group
            if seen_dict[src] > 0 or seen_dict[dst] > 0:
                for edge_index in edge_indices:
                    # Add two endpoints for these parallel edges
                    # so that they can be converted into line graph later
                    edges_src[edge_index] = max_value
                    edges_dst[edge_index] = max_value + 1
                max_value += 2
                seen_dict[src] += 1
                seen_dict[dst] += 1
            else:
                seen_dict[src] += 1
                seen_dict[dst] += 1
        graph_1 = dgl.graph((edges_src, edges_dst))

        edges_src = torch.from_numpy(df['SENDER_id'].to_numpy()).clone()
        edges_dst = torch.from_numpy(df['RECEIVER_id'].to_numpy()).clone()
        original_graph = dgl.graph((edges_src, edges_dst))
        edge_to_node_graph = self.edge_to_node_data_class()[0]["g"]  # Use this to derive a new graph
        graph_2, graph_3 = get_ppr_graph(original_graph, edge_to_node_graph, self.edge_to_node_data_folder,
                                         "transaction_edgetonode_bidirected", self.topppr_k, self.alpha)

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
        # for index, graph in enumerate(self.graphs):
        graph_1.edata["eid"] = torch.arange(edge_features.shape[0], dtype=torch.int)
        graph_1.edata["nfeat"] = edge_features
        graph_1.edata["train_mask"] = train_mask
        graph_1.edata["val_mask"] = val_mask
        graph_1.edata["test_mask"] = test_mask
        graph_1.edata["label"] = edge_labels
        # G = dgl.graph(([0,1,2],[1,2,1]))
        # get_line_graph(G, False)
        graph_1 = get_line_graph(graph_1, as_directed=False)

        self.graphs = [graph_1, graph_2, graph_3]
        for index, graph in enumerate(self.graphs):
            graph: dgl.DGLGraph
            l = graph.ndata["eid"]
            assert all(l[i] <= l[i + 1] for i in range(len(l) - 1))  # Make sure eids are sorted so they are aligned
            if self.add_self_loop:
                self.graphs[index] = graph.add_self_loop()
            else:  # Only add self loop for zero in degree nodes
                in_degrees = graph.in_degrees()
                node_index = torch.where(in_degrees == 0)[0]
                graph.add_edges(node_index, node_index)

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
#     dataset = Transaction3Sets(subset="train")
#     graph = dataset[0]
#     # graph = dgl.to_homogeneous(graph, ["user", "transaction"])
#     print(graph)