import pytorch_lightning as pl
from torch import nn
from dgl.nn.pytorch import GraphConv, SGConv, SAGEConv, GATConv, HeteroGraphConv, EdgeWeightNorm, RelGraphConv
import scipy.sparse as sp
import torch
import numpy as np
# pl.seed_everything(12)
from models.Base import BaseModel
from icecream import ic

from models.GNN import GNN


class GNN3Sets(BaseModel):
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
                 **kwargs):
        super(GNN3Sets, self).__init__(n_classes=n_classes)
        #         self.g = g
        self.n_classes = n_classes
        self.rel_names = rel_names
        self.target_node_type = target_node_type
        self.norm_edge_weight = nn.Parameter(torch.ones(1))  # Dummy parameter to be modified in forward
        self.gnns = nn.ModuleList()
        self.activation = activation
        if structure == "AGNN":
            self.last_layer = torch.nn.Linear(in_nfeats * 3, n_classes)
        else:
            self.last_layer = torch.nn.Linear(n_hidden * 3, n_classes)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        for _ in range(3):
            self.gnns.append(GNN(in_nfeats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout,
                 structure,
                 activation,
                 rel_names,
                 target_node_type,
                 pos_weight,
                 **kwargs))
        ic(self.gnns)

    def forward(self, g, nfeats, return_h_before_last_layer=False, **kwargs):
        # g here will be a list of gs
        hs = []

        for i, gnn in enumerate(self.gnns):
            h = gnn(g[i], nfeats, return_h_before_last_layer=True, **kwargs)
            hs.append(h)
        hs = torch.cat(hs, dim=1)
        if return_h_before_last_layer:
            return hs
        else:
            h = self.last_layer(hs)
        return h

    def _step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        labels = batch["labels"]
        nodes_mask = batch["nodes_mask"]
        if self.rel_names is None:
            logits = self(**batch).reshape(-1)
        else:
            logits = self(**batch)[self.target_node_type].reshape(-1)
            nodes_mask = nodes_mask[self.target_node_type]
            labels = labels[self.target_node_type]
        logits = logits[nodes_mask]
        labels = labels[nodes_mask]
        loss = self.loss_fn(logits, labels.float())
        logits = torch.sigmoid(logits)
        class_pred = (logits > 0.5).float()
        return {'loss': loss, "preds": logits, "pred_labels": class_pred, "labels": labels}

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        nodes_mask = batch["nodes_mask"]  # the train_mask
        logits = self(**batch, return_h_before_last_layer=False).reshape(-1)
        logits = logits[nodes_mask]
        labels = labels[nodes_mask]
        loss = self.loss_fn(logits, labels.float())
        logits = torch.sigmoid(logits)
        class_pred = (logits > 0.5).float()
        return {'loss': loss, "preds": logits, "pred_labels": class_pred, "labels": labels}

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
