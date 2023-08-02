import CommonModules as CM
import os

from datasets.base.Sets import Sets
from datasets.Yelpchi.YelpchiEdgeToNode import YelpchiEdgeToNode
from global_parameters import YelpchiEdgeToNodeDataFolder
import pandas as pd


class Yelpchi3Sets(Sets):
    # 38264 nodes, 67395 edges ->
    # directed edge to node graph for topk PPR
    # 67395 nodes, 67395 edges (with respect to self), 67395 nodes, 287619 edges (with respect to reviewer) and 67395
    # nodes, 486501 edges (with respect to product)
    # undirected edge to node graph for topk PPR
    # 67395 nodes, 67395 edges (with respect to self), 67395 nodes, 287619 edges (with respect to reviewer) and 67395
    # nodes, 176133 edges (with respect to product)
    def __init__(self, subset=None, no_node_features=False, homo=True, aggregate=False, bidirected=True,
                 initialized_instance=None, add_self_loop=True, topppr_k=1000, alpha=None):
        self.homo = homo
        self.aggregate = aggregate
        self.bidirected = bidirected
        if initialized_instance is None:
            self.df_train = pd.read_csv(os.path.join(YelpchiEdgeToNodeDataFolder, "train_edges.csv"))
            self.df_val = pd.read_csv(os.path.join(YelpchiEdgeToNodeDataFolder, "val_edges.csv"))
            self.df_test = pd.read_csv(os.path.join(YelpchiEdgeToNodeDataFolder, "test_edges.csv"))
            self.train_embeds, self.train_labels = CM.IO.ImportFromPkl(
                os.path.join(YelpchiEdgeToNodeDataFolder, "train_embeds_labels.pkl"))
            self.val_embeds, self.val_labels = CM.IO.ImportFromPkl(
                os.path.join(YelpchiEdgeToNodeDataFolder, "val_embeds_labels.pkl"))
            self.test_embeds, self.test_labels = CM.IO.ImportFromPkl(
                os.path.join(YelpchiEdgeToNodeDataFolder, "test_embeds_labels.pkl"))
        super().__init__(name='Yelpchi3Sets', edge_to_node_data_class=YelpchiEdgeToNode,
                         edge_to_node_data_folder=YelpchiEdgeToNodeDataFolder,
                         subset=subset, no_node_features=no_node_features,
                         homo=homo, aggregate=aggregate, bidirected=bidirected,
                         initialized_instance=initialized_instance, add_self_loop=add_self_loop,
                         topppr_k=topppr_k, alpha=alpha)


if __name__ == "__main__":
    dataset = Yelpchi3Sets(add_self_loop=False)
    graph = dataset[0]
    # graph = dgl.to_homogeneous(graph, ["user", "transaction"])
    print(graph)
