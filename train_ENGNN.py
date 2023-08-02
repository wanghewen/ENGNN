import torch

from dataloaders.GNN_dataloader import GNNDataloader
from datasets.Yelpchi.YelpchiENCombined import YelpchiENCombined
from global_parameters import wrapped_partial
from models.ENGNN import ENGNN
from train import main


def run(dataset_class, use_egnn=True, use_ngnn=True, use_3sets_linegraph=False, topppr_k=1000, alpha=None):
    if not use_ngnn:
        model_name = "OnlyEGNN"
    elif not use_egnn:
        model_name = "OnlyNGNN"
    else:
        model_name = "ENGNN"
    if use_3sets_linegraph:
        model_name += "_linegraph"
    dataset_class = wrapped_partial(dataset_class, topppr_k=topppr_k, alpha=alpha)

    logger_name = dataset_class.__name__+f"__GCN_{model_name}__k={topppr_k}__alpha={alpha}__baseline"
    use_pl = True
    model_parameter_dict = {"n_hidden": 128,
                            "n_classes": 1,
                            "n_layers": 2,
                            "activation": torch.nn.ReLU(),
                            "dropout": 0.5,
                            "structure": "GCN",
                            "node_count": dataset_class.get_node_count(),
                            "use_attention": False,
                            "use_egnn": use_egnn,
                            "use_ngnn": use_ngnn}
    dataloader_class = GNNDataloader
    model_class = ENGNN
    main(logger_name, dataset_class, dataloader_class, model_class, model_parameter_dict, use_pl)

    logger_name = dataset_class.__name__+f"__GraphSAGE_{model_name}__k={topppr_k}__alpha={alpha}__baseline"
    use_pl = True
    model_parameter_dict = {"n_hidden": 128,
                            "n_classes": 1,
                            "n_layers": 2,
                            "activation": torch.nn.ReLU(),
                            "dropout": 0.5,
                            "structure": "SAGE",
                            "aggregator_type": "mean",
                            "node_count": dataset_class.get_node_count(),
                            "use_attention": False,
                            "use_egnn": use_egnn,
                            "use_ngnn": use_ngnn}
    dataloader_class = GNNDataloader
    model_class = ENGNN
    main(logger_name, dataset_class, dataloader_class, model_class, model_parameter_dict, use_pl)

    logger_name = dataset_class.__name__+f"__GAT_{model_name}__k={topppr_k}__alpha={alpha}__baseline"
    use_pl = True
    model_parameter_dict = {"n_hidden": 128,
        "n_classes": 1,
        "n_layers": 2,
        "activation": torch.nn.ReLU(),
        "dropout": 0.5,
        "structure": "GAT",
        "node_count": dataset_class.get_node_count(),
                            "use_egnn": use_egnn,
                            "use_ngnn": use_ngnn}
    dataloader_class = GNNDataloader
    model_class = ENGNN
    main(logger_name, dataset_class, dataloader_class, model_class, model_parameter_dict, use_pl)

    logger_name = dataset_class.__name__ + f"__GIN_{model_name}__k={topppr_k}__alpha={alpha}__baseline"
    model_parameter_dict = {"n_hidden": 128,
                            "n_classes": 1,
                            "n_layers": 2,
                            "activation": torch.nn.ReLU(),
                            "dropout": 0.5,
                            "structure": "GIN",
                            "node_count": dataset_class.get_node_count(),
                            "aggregator_type": 'sum',
                            "use_egnn": use_egnn,
                            "use_ngnn": use_ngnn}
    dataloader_class = GNNDataloader
    model_class = ENGNN
    main(logger_name, dataset_class, dataloader_class, model_class, model_parameter_dict)

    logger_name = dataset_class.__name__ + f"__Cheb_{model_name}__k={topppr_k}__alpha={alpha}__baseline"
    model_parameter_dict = {"n_hidden": 128,
                            "n_classes": 1,
                            "n_layers": 2,
                            "activation": torch.nn.ReLU(),
                            "dropout": 0.5,
                            "structure": "Cheb",
                            "node_count": dataset_class.get_node_count(),
                            "use_egnn": use_egnn,
                            "use_ngnn": use_ngnn}
    dataloader_class = GNNDataloader
    model_class = ENGNN
    main(logger_name, dataset_class, dataloader_class, model_class, model_parameter_dict)

    logger_name = dataset_class.__name__ + f"__AGNN_{model_name}__k={topppr_k}__alpha={alpha}__baseline"
    model_parameter_dict = {"n_hidden": 128,
                            "n_classes": 1,
                            "n_layers": 2,
                            "activation": torch.nn.ReLU(),
                            "dropout": 0.5,
                            "structure": "AGNN",
                            "node_count": dataset_class.get_node_count(),
                            "use_egnn": use_egnn,
                            "use_ngnn": use_ngnn}
    dataloader_class = GNNDataloader
    model_class = ENGNN
    main(logger_name, dataset_class, dataloader_class, model_class, model_parameter_dict)

    logger_name = dataset_class.__name__ + f"__GATv2_{model_name}__k={topppr_k}__alpha={alpha}__baseline"
    model_parameter_dict = {"n_hidden": 128,
                            "n_classes": 1,
                            "n_layers": 2,
                            "activation": torch.nn.ReLU(),
                            "dropout": 0.5,
                            "structure": "GATv2",
                            "node_count": dataset_class.get_node_count(),
                            "use_egnn": use_egnn,
                            "use_ngnn": use_ngnn}
    dataloader_class = GNNDataloader
    model_class = ENGNN
    main(logger_name, dataset_class, dataloader_class, model_class, model_parameter_dict)



if __name__ == "__main__":
    run(YelpchiENCombined)


