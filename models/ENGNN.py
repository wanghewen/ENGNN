from torch import nn
import dgl.function as fn
import torch
import numpy as np
from models.Base import BaseModel
from icecream import ic

from models.GNN3Sets import GNN3Sets
from models.GNNEdgeAttributed import GNNEdgeAttributed
from math import sqrt


class SelfAttention(nn.Module):
    dim_in: int
    dim_k: int
    dim_v: int

    def __init__(self, dim_in, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        # x: batch, n, dim_in
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = x  # batch, n, dim_v

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n, n
        att = torch.bmm(dist, v)
        return att, dist


class ENGNN(BaseModel):
    def __init__(self,
                 #                  g,
                 in_nfeats,
                 # in_efeats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout,
                 structure,
                 activation=torch.nn.ReLU(),
                 rel_names=None,
                 target_node_type=None,
                 pos_weight=None,
                 node_count=None,
                 use_attention=False,
                 use_egnn=True,
                 use_ngnn=True,
                 **kwargs):
        super(ENGNN, self).__init__(n_classes=n_classes)
        self.n_classes = n_classes
        self.rel_names = rel_names
        self.target_node_type = target_node_type
        self.use_attention = use_attention
        self.use_egnn = use_egnn
        self.use_ngnn = use_ngnn
        self.gnns = nn.ModuleList()
        self.activation = activation
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if self.use_egnn:
            self.gnns.append(GNN3Sets(in_nfeats,
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
        if structure == "AGNN":
            self.middle_batch_norm = torch.nn.BatchNorm1d(in_nfeats * 3)
            self.middle_layer = torch.nn.Linear(in_nfeats * 3, in_nfeats)
        else:
            self.middle_batch_norm = torch.nn.BatchNorm1d(n_hidden * 3)
            self.middle_layer = torch.nn.Linear(n_hidden * 3, in_nfeats)
        if self.use_ngnn:
            self.gnns.append(GNNEdgeAttributed(in_nfeats,
                                               in_nfeats,
                                               n_hidden,
                                               n_classes,
                                               n_layers,
                                               dropout,
                                               structure,
                                               activation,
                                               target_node_type,
                                               pos_weight,
                                               no_node_features=False,
                                               node_count=node_count,
                                               **kwargs))
        if structure == "AGNN":
            self.padded_size = in_nfeats * 3
        else:
            self.padded_size = max(n_hidden * 3, in_nfeats + n_hidden * 2)
        # self.attention_layer = torch.nn.MultiheadAttention(self.padded_size, 8, batch_first=True, dropout=0.5, kdim=10)
        if self.use_attention:
            self.attention_layer = SelfAttention(self.padded_size, 1, self.padded_size)

        last_layer_input_size = self.padded_size

        self.batch_norm = torch.nn.BatchNorm1d(last_layer_input_size)
        self.last_layer = torch.nn.Linear(last_layer_input_size, n_classes)
        # self.last_layer = torch.nn.Linear(2, n_classes)
        ic(self.gnns)

    def forward(self, g, return_h_before_last_layer=False, **kwargs):
        # g here will be a list of gs
        hs = []
        target_edge_count = kwargs["nfeats"].shape[0]
        # EGNN
        if self.use_egnn:
            h = self.gnns[0](g[0], return_h_before_last_layer=True, **kwargs)
        else:
            h = kwargs["nfeats"]
        h = self.middle_batch_norm(h)
        h1 = torch.nn.functional.pad(h, (0, self.padded_size - h.shape[1]))
        hs.append(h1[:target_edge_count].unsqueeze(1))  # Use mask to drop self loop for gnns[1]'s output
        h = self.middle_layer(h)

        # NGNN
        if self.use_ngnn:
            g[1].edata["efeat"][:h.shape[0]] = h
            g[1].update_all(fn.copy_e('efeat', 'm'), fn.mean('m', 'nfeat'))
            kwargs["nfeats"] = g[1].ndata["nfeat"]
            h = self.gnns[1](g[1], return_h_before_last_layer=True, **kwargs)
            h1 = torch.nn.functional.pad(h, (0, self.padded_size - h.shape[1]))
            hs.append(h1[:target_edge_count].unsqueeze(1))  # Use mask to drop self loop for gnns[1]'s output

        hs = torch.cat(hs, dim=1)
        if self.use_attention:
            hs, hs_weights = self.attention_layer(hs)
            ic(hs_weights)
        # hs = hs_weights * hs
        hs = hs.sum(1)
        # hs = hs.mean(1)
        hs = self.batch_norm(hs)
        if return_h_before_last_layer:
            return hs
        else:
            h = self.last_layer(hs)

        return h

    def _step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        labels = batch["labels"]
        mask = batch["mask"]
        if self.rel_names is None:
            logits = self(**batch).reshape(-1)
        else:
            logits = self(**batch)[self.target_node_type].reshape(-1)
            mask = mask[self.target_node_type]
            labels = labels[self.target_node_type]
        logits = logits[mask]
        labels = labels[mask]
        loss = self.loss_fn(logits, labels.float())
        logits = torch.sigmoid(logits)
        class_pred = (logits > 0.5).float()
        return {'loss': loss, "preds": logits, "pred_labels": class_pred, "labels": labels}

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        mask = batch["mask"]  # the train_mask
        logits = self(**batch, return_h_before_last_layer=False).reshape(-1)
        logits = logits[mask]
        labels = labels[mask]
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
