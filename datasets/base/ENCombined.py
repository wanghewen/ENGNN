from dgl.data import DGLDataset
import torch

from icecream import ic

# Refer to https://docs.dgl.ai/en/latest/new-tutorial/6_load_data.html
# And https://github.com/minhduc0711/gcn/blob/2d09dd2f5c5400e4b53f74e2b35a5a612904ae48/src/data/datasets.py
class ENCombined(DGLDataset):
    # 38264 nodes, 67395 edges ->
    @classmethod
    def get_node_count(cls):
        raise NotImplementedError

    def __init__(self, name, dataclass_3sets, dataclass_edge_attributed, subset=None, no_node_features=False,
                 initialized_instance=None):
        # self.homo = homo
        if initialized_instance is None:
            super().__init__(name=name)
            self.e_graph_dataset = dataclass_3sets(subset=subset,
                                                initialized_instance=None)
            self.n_graph_dataset = dataclass_edge_attributed(subset=subset,
                                                         initialized_instance=None)
            self.edges_id = torch.arange(self.n_graph_dataset.graph.number_of_edges())
        else:
            self = initialized_instance
        self.e_graphs = self.e_graph_dataset.graphs  # 3 Sets Graph
        self.n_graph = self.n_graph_dataset.graph  # Original Graph
        if subset == "train":
            self.e_graph_node_mask = self.e_graphs[0].ndata["train_mask"]
            # Filter out self loops using e_graph_node_mask
            self.n_graph_edge_mask = self.n_graph.edata["train_mask"][:self.e_graph_node_mask.shape[0]]
        elif subset == "val":
            self.e_graph_node_mask = self.e_graphs[0].ndata["val_mask"]
            self.n_graph_edge_mask = self.n_graph.edata["val_mask"][:self.e_graph_node_mask.shape[0]]
        elif subset == "test":
            self.e_graph_node_mask = self.e_graphs[0].ndata["test_mask"]
            self.n_graph_edge_mask = self.n_graph.edata["test_mask"][:self.e_graph_node_mask.shape[0]]
        else:
            self.e_graph_node_mask = None
            self.n_graph_edge_mask = None
        if no_node_features:  # Exclude any node features
            # self.nfeats = torch.eye(self.graph.num_nodes())
            # self.nfeats = torch.nn.Parameter(torch.rand(*self.graph.ndata.shape))
            self.nfeats_e_graphs = None
        else:
            self.nfeats_e_graphs = self.e_graphs[0].ndata["nfeat"]

        self.efeats_n_graph = self.n_graph.edata["efeat"]

    def process(self):
        pass

    def __getitem__(self, i):
        return {
            "g": [self.e_graphs, self.n_graph],
            "nfeats": self.nfeats_e_graphs,
            "efeats": self.efeats_n_graph,
            "labels": self.e_graphs[0].ndata["label"],
            "mask": self.e_graph_node_mask,
            "edges_id": self.edges_id,
            # "edges_src": self.edges_src,
            # "edges_dst": self.edges_dst
        }

    def __len__(self):
        return 1
