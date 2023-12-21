import os

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
import CommonModules as CM


class GNN(BaseModel):
    def __init__(self,
                 in_nfeats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout,
                 structure,
                 activation=torch.nn.ReLU(),
                 rel_names=None,
                 target_node_type=None,
                 pos_weight=None,
                 random_adj=False,
                 adj_method=None,
                 use_gb=False,
                 **kwargs):
        super(GNN, self).__init__(n_classes=n_classes)
        #         self.g = g
        self.n_classes = n_classes
        self.rel_names = rel_names
        self.target_node_type = target_node_type
        assert random_adj in [False, "once", "every_epoch", "every_epoch_double_norm"]
        self.random_adj = random_adj
        self.randomed_adj_flag = False
        self.adj_method = adj_method
        self.smote_adj = None
        self.use_gb = use_gb

        # Dummy parameter to be modified in forward.
        self.norm_edge_weight = nn.Parameter(torch.ones(1))
        self.layers = nn.ModuleList()
        #         self.last_layer = nn.Linear(n_hidden*2+in_efeats, n_classes) # Use this one to concat edge attrs and get labels
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
            self.layers.append(nn.Dropout(p=dropout))
            for i in range(n_layers - 1):
                self.layers.append(torch.nn.Linear(n_hidden, n_hidden))
                self.layers.append(torch.nn.ReLU())
                self.layers.append(nn.Dropout(p=dropout))
        elif structure == "RGCN":
            #             self.layers.append(HeteroGraphConv({rel: GraphConv(in_nfeats, n_hidden, activation=activation) for rel in self.rel_names}, aggregate='sum'))
            self.layers.append(
                RelGraphConv(in_nfeats, n_hidden, len(self.rel_names), activation=activation, low_mem=True,
                             dropout=dropout))
            # hidden layers
            for i in range(n_layers - 1):
                #                 self.layers.append(HeteroGraphConv({rel: GraphConv(n_hidden, n_hidden, activation=activation) for rel in self.rel_names}, aggregate='sum'))
                self.layers.append(
                    RelGraphConv(n_hidden, n_hidden, len(self.rel_names), activation=activation, low_mem=True,
                                 dropout=dropout))
        if structure == "AGNN":
            last_layer_input_size = in_nfeats
        else:
            last_layer_input_size = n_hidden
        if self.use_gb:
            self.last_layer = torch.nn.Sequential(torch.nn.BatchNorm1d(last_layer_input_size), torch.nn.Linear(last_layer_input_size, n_classes))
        else:
            self.last_layer = torch.nn.Linear(last_layer_input_size, n_classes)
        ic(self.layers)

    def forward(self, g, nfeats, return_h_before_last_layer=False, **kwargs):
        h = nfeats
        for i, layer in enumerate(self.layers):
            if type(layer) == torch.nn.Embedding:
                h = layer()
            elif type(layer) == torch.nn.modules.dropout.Dropout:
                if type(h) == dict:  # RGCN case
                    h = {ntype: layer(_h) for ntype, _h in h.items()}
                else:
                    h = layer(h)
            elif type(layer) in [torch.nn.Tanh, torch.nn.ReLU]:
                h = layer(h)
            elif type(layer) == torch.nn.Linear:
                h = layer(h)
                layer: torch.nn.Linear
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
                h = layer(g, h)
            else:
                h = layer(g, h)
        h_before_last_layer = h
        if return_h_before_last_layer:
            return h_before_last_layer
        else:
            h = self.last_layer(h_before_last_layer)
            return h

    def _step(self, batch, batch_idx):
        labels = batch["g"].ndata["label"]
        nodes_mask = batch["nodes_mask"]
        if self.rel_names is None:
            logits = self(**batch).reshape(-1, self.n_classes)
        else:
            logits = self(**batch)[self.target_node_type].reshape(-1, self.n_classes)
            nodes_mask = nodes_mask[self.target_node_type]
            labels = labels[self.target_node_type]
        logits = logits[nodes_mask]
        labels = labels[nodes_mask]
        loss = self.loss_fn(logits, labels.float())
        logits = torch.sigmoid(logits)
        class_pred = (logits > 0.5).float()
        return {'loss': loss, "preds": logits.detach(), "pred_labels": class_pred.detach(), "labels": labels.detach()}

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        batch["nodes_mask"] = batch.get("train_mask", batch["nodes_mask"])  # update train_mask if possible
        nodes_mask = batch["nodes_mask"]
        if self.rel_names is None:
            h_before_last_layer = self(**batch, return_h_before_last_layer=True)
        else:
            h_before_last_layer = self(**batch, return_h_before_last_layer=True)

        labels = labels[nodes_mask]
        logits = self.last_layer(h_before_last_layer).reshape(-1, self.n_classes)
        logits = logits[nodes_mask]
        loss = self.loss_fn(logits, labels.float())
        logits = torch.sigmoid(logits)
        class_pred = (logits > 0.5).float()
        return {'loss': loss, "preds": logits.detach(), "pred_labels": class_pred.detach(), "labels": labels.detach()}

    def training_epoch_end(self, output):
        # log epoch metric
        preds = torch.cat([x['preds'] for x in output])
        pred_labels = torch.cat([x['pred_labels'] for x in output])
        labels = torch.cat([x['labels'] for x in output])
        loss = torch.stack([x['loss'] for x in output]).detach().mean()

        self.log_result(preds, pred_labels, labels, loss, dataset="train", logs_to_logger=True)
        pass

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, batch_idx)  # Same code

    def validation_epoch_end(self, output):
        # log epoch metric
        val_output = output[0]
        preds = torch.cat([x['preds'] for x in val_output]).detach()
        pred_labels = torch.cat([x['pred_labels'] for x in val_output]).detach()
        labels = torch.cat([x['labels'] for x in val_output]).detach()
        loss = torch.stack([x['loss'] for x in val_output]).detach().mean()
        self.log_result(preds, pred_labels, labels, loss, dataset="val", logs_to_logger=True)
        if len(output) > 1:  # There exists a test dataloader
            test_output = output[1]
            preds = torch.cat([x['preds'] for x in test_output]).detach()
            pred_labels = torch.cat([x['pred_labels'] for x in test_output]).detach()
            labels = torch.cat([x['labels'] for x in test_output]).detach()
            loss = torch.stack([x['loss'] for x in test_output]).detach().mean()
            self.log_result(preds, pred_labels, labels, loss, dataset="test_run", logs_to_logger=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, batch_idx)  # Same code

    def test_epoch_end(self, output):
        # log epoch metric
        preds = torch.cat([x['preds'] for x in output])
        pred_labels = torch.cat([x['pred_labels'] for x in output])
        labels = torch.cat([x['labels'] for x in output])
        loss = torch.stack([x['loss'] for x in output]).detach().mean()

        self.log_result(preds, pred_labels, labels, loss, dataset="test", logs_to_logger=True)
