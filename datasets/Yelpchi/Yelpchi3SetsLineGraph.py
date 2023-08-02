import CommonModules as CM
import os

from datasets.base.SetsLineGraph import SetsLineGraph
from global_parameters import YelpchiEdgeToNodeDataFolder
import pandas as pd
from icecream import ic


class Yelpchi3SetsLineGraph(SetsLineGraph):
    # 38264 nodes, 67395 edges -> 67395 nodes, 67395 edges (with respect to self), 67395 nodes, 48223145 edges (with
    # respect to product) or 67395 nodes, 287619 edges (with respect to reviewer)
    # (as undirected graph)
    def __init__(self, subset=None, no_node_features=False, homo=True, aggregate=False, bidirected=True,
                 initialized_instance=None):
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
        super().__init__(name='Yelpchi3Sets_LineGraph', subset=subset, no_node_features=no_node_features, homo=homo,
                         aggregate=aggregate, bidirected=bidirected, initialized_instance=initialized_instance)


if __name__ == "__main__":
    dataset = Yelpchi3SetsLineGraph()
    graph = dataset[0]
    # graph = dgl.to_homogeneous(graph, ["user", "transaction"])
    print(graph)
