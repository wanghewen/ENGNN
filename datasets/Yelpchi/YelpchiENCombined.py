from datasets.Yelpchi.Yelpchi3Sets import Yelpchi3Sets
from datasets.Yelpchi.Yelpchi3SetsLineGraph import Yelpchi3SetsLineGraph
from datasets.Yelpchi.YelpchiEdgeAttributed import YelpchiEdgeAttributed
from datasets.base.ENCombined import ENCombined
from global_parameters import wrapped_partial


# Refer to https://docs.dgl.ai/en/latest/new-tutorial/6_load_data.html
# And https://github.com/minhduc0711/gcn/blob/2d09dd2f5c5400e4b53f74e2b35a5a612904ae48/src/data/datasets.py
class YelpchiENCombined(ENCombined):
    # 38264 nodes, 67395 edges ->
    @classmethod
    def get_node_count(cls):
        return 38264

    def __init__(self, subset=None, no_node_features=False, initialized_instance=None, use_3sets_linegraph=False,
                 topppr_k=1000, alpha=None):
        if use_3sets_linegraph:
            dataclass_3sets = Yelpchi3SetsLineGraph
        else:
            dataclass_3sets = wrapped_partial(Yelpchi3Sets, topppr_k=topppr_k, alpha=alpha)
        self.use_3sets_linegraph = use_3sets_linegraph
        super().__init__(name="YelpchiENCombined", dataclass_3sets=dataclass_3sets,
                         dataclass_edge_attributed=YelpchiEdgeAttributed,
                         subset=subset,
                         no_node_features=no_node_features, initialized_instance=initialized_instance)


if __name__ == "__main__":
    dataset = YelpchiENCombined(subset="train")
    graph = dataset[0]
    # graph = dgl.to_homogeneous(graph, ["user", "transaction"])
    print(graph)