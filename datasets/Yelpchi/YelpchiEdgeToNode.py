import CommonModules as CM
import os
import pandas as pd

from datasets.base.EdgeToNode import EdgeToNode
from global_parameters import YelpchiEdgeToNodeDataFolder
from icecream import ic

# Refer to https://docs.dgl.ai/en/latest/new-tutorial/6_load_data.html
# And https://github.com/minhduc0711/gcn/blob/2d09dd2f5c5400e4b53f74e2b35a5a612904ae48/src/data/datasets.py
class YelpchiEdgeToNode(EdgeToNode):
    # 38264 nodes, 67395 edges -> 105659 nodes, 375239 edges
    @classmethod
    def get_augment_enode_count(cls):
        """
        Upper limit for initialize augmented enode features. Must be greater than or equal to the number of positive nodes in the training set.
        :return:
        """
        return 5639*2

    def __init__(self, subset=None, no_node_features=False, homo=True, aggregate=False, initialized_instance=None,
                 randomize_train_test=False, bidirectional=True, end_node_same_type=True, randomize_by_node=False,
                 **kwargs):
        data_folder = YelpchiEdgeToNodeDataFolder
        super().__init__(name="YelpchiEdgeToNode", data_folder=data_folder, subset=subset,
                         no_node_features=no_node_features, homo=homo, aggregate=aggregate,
                         initialized_instance=initialized_instance, randomize_train_test=randomize_train_test,
                         bidirectional=bidirectional, end_node_same_type=end_node_same_type, 
                         randomize_by_node=randomize_by_node, **kwargs)


if __name__ == "__main__":
    dataset = YelpchiEdgeToNode(subset="train")
    graph = dataset[0]
    # graph = dgl.to_homogeneous(graph, ["user", "transaction"])
    print(graph)