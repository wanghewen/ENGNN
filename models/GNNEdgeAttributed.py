import dgl
import pytorch_lightning as pl
from torch import nn
from dgl.nn.pytorch import GraphConv, SGConv, SAGEConv, GATConv, HeteroGraphConv, EdgeWeightNorm, RelGraphConv, GINConv, \
    PNAConv, ChebConv, AGNNConv, GATv2Conv, HGTConv
import scipy.sparse as sp
import torch
import numpy as np
# pl.seed_everything(12)
from models.Base import BaseModel
from icecream import ic


class GNNEdgeAttributed(BaseModel):
    def __init__(self,
                 in_nfeats,
                 in_efeats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout,
                 structure,
                 activation=torch.nn.ReLU(),
                 target_node_type=None,
                 pos_weight=None,
                 use_smote=False,
                 random_adj=False,
                 adj_method=None,
                 feature_augment=False,
                 no_node_features=False,
                 node_count=None,
                 **kwargs):
        super(GNNEdgeAttributed, self).__init__(n_classes=n_classes)
        #         self.g = g
        self.n_classes = n_classes
        # self.rel_names = rel_names
        self.target_node_type = target_node_type
        self.use_smote = use_smote
        assert random_adj in [False, "once", "every_epoch", "every_epoch_double_norm"]
        self.random_adj = random_adj
        self.randomed_adj_flag = False
        self.adj_method = adj_method
        self.feature_augment = feature_augment
        self.smote_adj = None
        self.norm_edge_weight = nn.Parameter(torch.ones(1))  # Dummy parameter to be modified in forward
        self.layers = nn.ModuleList()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if no_node_features:
            self.layers.append(nn.Embedding(node_count, in_nfeats))
        if structure == "GCN":
            # input layer
            if random_adj in ["once", "every_epoch"]:
                self.layers.append(GraphConv(in_nfeats, n_hidden, activation=activation, norm='none'))
            else:
                self.layers.append(GraphConv(in_nfeats, n_hidden, activation=activation))
            # hidden layers
            for i in range(n_layers - 1):
                if random_adj in ["once", "every_epoch"]:
                    self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation, norm='none'))
                else:
                    self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
                self.layers.append(nn.Dropout(p=dropout))
        elif structure == "SGC":
            self.layers.append(SGConv(in_nfeats, n_hidden, k=n_layers, cached=True, bias=True, norm=None))
        elif structure == "GAT":
            self.layers.append(GATConv(in_nfeats, n_hidden, num_heads=1, feat_drop=dropout, activation=activation))
            # hidden layers
            for i in range(n_layers - 1):
                self.layers.append(GATConv(n_hidden, n_hidden, num_heads=1, feat_drop=dropout, activation=activation))
        elif structure == "SAGE":
            # input layer
            aggregator_type = kwargs["aggregator_type"]
            if random_adj in ["once", "every_epoch"]:
                self.layers.append(
                    SAGEConv(in_nfeats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation, norm=None))
            else:
                self.layers.append(
                    SAGEConv(in_nfeats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
            # hidden layers
            for i in range(n_layers - 1):
                if random_adj in ["once", "every_epoch"]:
                    self.layers.append(
                        SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation,
                                 norm=None))
                else:
                    self.layers.append(
                        SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        elif structure == "GIN":
            aggregator_type = kwargs["aggregator_type"]
            lin = torch.nn.Linear(in_nfeats, n_hidden)
            self.layers.append(GINConv(lin, aggregator_type, activation=activation))
            # hidden layers
            for i in range(n_layers - 1):
                lin = torch.nn.Linear(n_hidden, n_hidden)
                self.layers.append((GINConv(lin, aggregator_type, activation=activation)))
        elif structure == "PNA":
            aggregator_type = kwargs["aggregator_type"]
            scaler_type = kwargs["scaler_type"]
            self.layers.append(PNAConv(in_nfeats, n_hidden, aggregator_type, scalers=scaler_type,
                                       delta=2, dropout=dropout))
            # hidden layers
            for i in range(n_layers - 1):
                self.layers.append((PNAConv(n_hidden, n_hidden, aggregator_type, scalers=scaler_type,
                                    delta=2, dropout=dropout)))
        elif structure == "Cheb":
            self.layers.append(ChebConv(in_nfeats, n_hidden, 2, activation=activation))
            # hidden layers
            for i in range(n_layers - 1):
                self.layers.append(ChebConv(n_hidden, n_hidden, 2, activation=activation))
        elif structure == "AGNN":
            self.layers.append(AGNNConv())
            # hidden layers
            for i in range(n_layers - 1):
                self.layers.append(AGNNConv())
        elif structure == "GATv2":
            self.layers.append(GATv2Conv(in_nfeats, n_hidden, num_heads=1, feat_drop=dropout, activation=activation))
            # hidden layers
            for i in range(n_layers - 1):
                self.layers.append(GATv2Conv(n_hidden, n_hidden, num_heads=1, feat_drop=dropout, activation=activation))
        elif structure == "Transformer":
            self.layers.append(HGTConv(in_nfeats, n_hidden, num_heads=1, dropout=dropout, num_ntypes=1, num_etypes=1))
            self.layers.append(torch.nn.ReLU())
            # hidden layers
            for i in range(n_layers - 1):
                self.layers.append(HGTConv(n_hidden, n_hidden, num_heads=1, dropout=dropout, num_ntypes=1, num_etypes=1))
                self.layers.append(torch.nn.ReLU())
        elif structure == "FC":
            self.layers.append(torch.nn.Linear(in_nfeats, n_hidden))
            self.layers.append(torch.nn.ReLU())
            for i in range(n_layers - 1):
                self.layers.append(torch.nn.Linear(n_hidden, n_hidden))
                self.layers.append(torch.nn.Tanh())
        elif structure == "RGCN":
            self.layers.append(
                RelGraphConv(in_nfeats, n_hidden, len(self.rel_names), activation=activation, low_mem=True,
                             dropout=dropout))
            # hidden layers
            for i in range(n_layers - 1):
                self.layers.append(
                    RelGraphConv(n_hidden, n_hidden, len(self.rel_names), activation=activation, low_mem=True,
                                 dropout=dropout))
        if structure == "AGNN":
            self.last_layer = nn.Linear(in_nfeats * 2 + in_efeats,
                                        n_classes)  # Use this one to concat edge attrs and get labels
        else:
            self.last_layer = nn.Linear(n_hidden * 2 + in_efeats,
                                        n_classes)  # Use this one to concat edge attrs and get labels
        ic(self.layers)

    def forward(self, g: dgl.DGLGraph, nfeats, efeats, edges_id, return_h_before_last_layer=False, **kwargs):
        h = nfeats
        edges_src, edges_dst = g.find_edges(edges_id)
        for i, layer in enumerate(self.layers):
            #             print(layer)
            if type(layer) == torch.nn.Embedding:
                h = layer(torch.arange(g.number_of_nodes()).to(g.device))
            elif type(layer) == torch.nn.modules.dropout.Dropout:
                if type(h) == dict:  # RGCN case
                    h = {ntype: layer(_h) for ntype, _h in h.items()}
                else:
                    h = layer(h)
            elif type(layer) in [torch.nn.Tanh, torch.nn.ReLU]:
                h = layer(h)
            elif type(layer) == torch.nn.Linear:
                h = layer(h)
            elif type(layer) in [GATConv, GATv2Conv]:
                h = layer(g, h).mean(1)
            elif type(layer) == HGTConv:
                h = layer(g, h, torch.zeros(g.number_of_nodes(), dtype=torch.long).to(g.device),
                          torch.zeros(g.number_of_edges(), dtype=torch.long).to(g.device))
            elif type(layer) == HeteroGraphConv:
                h = layer(g, h)
            elif type(layer) == RelGraphConv:
                h = layer(g, h, g.edata["_TYPE"])
            elif type(layer) in [GraphConv, SAGEConv]:
                if self.random_adj in ["once", "every_epoch", "every_epoch_double_norm"]:
                    if (not self.randomed_adj_flag) or self.random_adj in ["every_epoch", "every_epoch_double_norm"]:
                        #                         print(self.training, self.norm_edge_weight)
                        if self.training or self.norm_edge_weight.shape == (1,):
                            edge_weight = torch.rand(g.number_of_edges(), device=h.device) + torch.finfo(
                                torch.float32).eps  # Avoid 0 edge weight
                            norm = EdgeWeightNorm(norm='both')
                            self.norm_edge_weight = nn.Parameter(norm(g, edge_weight))  # Keep last edge weight
                            self.randomed_adj_flag = True
                    h = layer(g, h, edge_weight=self.norm_edge_weight)
                else:
                    h = layer(g, h)
            else:
                h = layer(g, h)
        h_before_last_layer = h
        h_before_last_layer = torch.cat([h_before_last_layer[edges_src], h_before_last_layer[edges_dst],
                                         efeats[edges_id]], dim=1)
        if return_h_before_last_layer:
            return h_before_last_layer
        else:
            h = self.last_layer(h_before_last_layer)
            return h

    def _step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        labels = batch["labels"]
        edges_mask = batch["edges_mask"]
        logits = self(**batch)
        logits = logits[edges_mask]
        labels = labels[edges_mask]
        loss = self.loss_fn(logits, labels.float())
        logits = torch.sigmoid(logits)
        class_pred = (logits > 0.5).float()
        return {'loss': loss, "preds": logits, "pred_labels": class_pred, "labels": labels}

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)  # Same code


    def training_epoch_end(self, output):
        # log epoch metric
        preds = torch.cat([x['preds'] for x in output]).detach()
        pred_labels = torch.cat([x['pred_labels'] for x in output]).detach()
        labels = torch.cat([x['labels'] for x in output]).detach()
        loss = torch.stack([x['loss'] for x in output]).detach().mean()

        self.log_result(preds, pred_labels, labels, loss, dataset="train")
        pass

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, batch_idx)  # Same code

    def validation_epoch_end(self, output):
        # log epoch metric
        output = output[0]
        preds = torch.cat([x['preds'] for x in output]).detach()
        pred_labels = torch.cat([x['pred_labels'] for x in output]).detach()
        labels = torch.cat([x['labels'] for x in output]).detach()
        loss = torch.stack([x['loss'] for x in output]).detach().mean()
        self.log_result(preds, pred_labels, labels, loss, dataset="val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, batch_idx)  # Same code

    def test_epoch_end(self, output):
        # log epoch metric
        preds = torch.cat([x['preds'] for x in output]).detach()
        pred_labels = torch.cat([x['pred_labels'] for x in output]).detach()
        labels = torch.cat([x['labels'] for x in output]).detach()
        loss = torch.stack([x['loss'] for x in output]).detach().mean()

        self.log_result(preds, pred_labels, labels, loss, dataset="test")
