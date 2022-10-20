# !/usr/bin/env python
# -*-coding:utf-8 -*-
# File       : model.py
# Author     ：Clark Wang
# version    ：python 3.x
import glob
import torch
import random
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from torch import nn
from torch.nn import functional
from torch_geometric.nn import CGConv, GATv2Conv, SAGEConv, SuperGATConv
from GNNLayers import SelectGAT

class models(torch.nn.Module):

    def __init__(self, args):
        super(models, self).__init__()
        self.args = args
        self.in_channels = args.feature_length
        self.device = args.device
        filters = self.args.filters.split('_')
        self.gcn_filters = [int(n_filter) for n_filter in filters]
        self.gcn_numbers = len(self.gcn_filters)
        self.gcn_last_filter = self.gcn_filters[-1]

        gcn_parameters = [dict(in_channels=self.gcn_filters[i - 1], out_channels=self.gcn_filters[i], bias=True) for i
                          in range(1, self.gcn_numbers)]
        gcn_parameters.insert(0, dict(in_channels=self.in_channels, out_channels=self.gcn_filters[0], bias=True))

        self.conv_layer_dict = {
            "gcn": dict(constructor=CGConv, kwargs=gcn_parameters),
            "gat": dict(constructor=GATv2Conv, kwargs=gcn_parameters),
            "supergat": dict(constructor=SuperGATConv, kwargs=gcn_parameters),
            "sage": dict(constructor=SAGEConv, kwargs=gcn_parameters),
            "selectgat": dict(constructor=SelectGAT, kwargs=gcn_parameters)
        }
        self.setup_layers()


    def setup_layers(self):
        conv = self.conv_layer_dict[self.args.conv]
        constructor = conv['constructor']
        setattr(self, 'gc{}'.format(1), constructor(**conv['kwargs'][0]))
        for i in range(1, self.gcn_numbers):
            setattr(self, 'gc{}'.format(i + 1), constructor(**conv['kwargs'][i]))

        ### global
        if self.args.global_agg.lower() == 'fc_max_pool':
            self.global_fc_agg = nn.Linear(self.gcn_last_filter, self.gcn_last_filter)
        elif self.args.global_agg.lower() == 'fc_mean_pool':
            self.global_fc_agg = nn.Linear(self.gcn_last_filter, self.gcn_last_filter)
        elif self.args.global_agg.lower() == 'lstm':
            self.global_lstm_agg = nn.LSTM(input_size=self.gcn_last_filter, hidden_size=self.gcn_last_filter,
                                           num_layers=1, bidirectional=True, batch_first=True)
        else:
            raise NotImplementedError

        ### Node-Graph
        self.perspectives = self.args.perspectives
        if self.args.match.lower() == 'node-graph':
            self.mp_w = nn.Parameter(torch.rand(self.perspectives,
                                                self.gcn_last_filter))  # trainable weight matrix for multi-perspective matching function
            self.lstm_input_size = self.perspectives
        else:
            raise NotImplementedError

        ### Agg layer
        self.hidden_size = self.args.hidden_size
        if self.args.match_agg.lower() == 'bilstm':
            self.agg_bilstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.hidden_size, num_layers=1,
                                      bidirectional=True, batch_first=True)
        elif self.args.match_agg.lower() == 'fc_avg' or self.args.match_agg.lower() == 'fc_max':
            self.fc_agg = nn.Linear(self.lstm_input_size, self.lstm_input_size)
        elif self.args.match_agg.lower() == 'avg' or self.args.match_agg.lower() == 'max':
            pass
        else:
            raise NotImplementedError

        ### Predict
        if self.args.task.lower() == 'regression':
            if self.global_flag is True:
                if self.global_agg.lower() == 'lstm':
                    factor_global = 2
                else:
                    factor_global = 1
            else:
                factor_global = 0
            if self.args.match_agg == 'bilstm':
                factor_match_agg = 2
            else:
                factor_match_agg = 1
            factor = factor_match_agg + factor_global
            self.predict_fc1 = nn.Linear(int(self.hidden_size * 2 * factor), int(self.hidden_size * factor))
            self.predict_fc2 = nn.Linear(int(self.hidden_size * factor), int((self.hidden_size * factor) / 2))
            self.predict_fc3 = nn.Linear(int((self.hidden_size * factor) / 2), int((self.hidden_size * factor) / 4))
            self.predict_fc4 = nn.Linear(int((self.hidden_size * factor) / 4), 1)
        elif self.args.task.lower() == 'classification':
            print("classification task")
        else:
            raise NotImplementedError

    def global_aggregation_info(self, v, agg_func_name):
        """
        :param v: (batch, len, dim)
        :param agg_func_name:
        :return: (batch, len)
        """
        if agg_func_name.lower() == 'max_pool':
            agg_v = torch.max(v, 1)[0]
        elif agg_func_name.lower() == 'fc_max_pool':
            agg_v = self.global_fc_agg(v)
            agg_v = torch.max(agg_v, 1)[0]
        elif agg_func_name.lower() == 'mean_pool':
            agg_v = torch.mean(v, dim=1)
        elif agg_func_name.lower() == 'fc_mean_pool':
            agg_v = self.global_fc_agg(v)
            agg_v = torch.mean(agg_v, dim=1)
        elif agg_func_name.lower() == 'lstm':
            _, (agg_v_last, _) = self.global_lstm_agg(v)
            agg_v = agg_v_last.permute(1, 0, 2).contiguous().view(-1, self.gcn_last_filter * 2)
        else:
            raise NotImplementedError
        return agg_v

    @staticmethod
    def div_with_small_value(n, d, eps=1e-8):
        # too small values are replaced by 1e-8 to prevent it from exploding.
        d = d * (d > eps).float() + eps * (d <= eps).float()
        return n / d

    def cosine_attention(self, v1, v2):
        """
        :param v1: (batch, len1, dim)
        :param v2: (batch, len2, dim)
        :return:  (batch, len1, len2)
        """
        # (batch, len1, len2)
        a = torch.bmm(v1, v2.permute(0, 2, 1))

        v1_norm = v1.norm(p=2, dim=2, keepdim=True)  # (batch, len1, 1)
        v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)  # (batch, len2, 1)
        d = v1_norm * v2_norm
        return self.div_with_small_value(a, d)

    def multi_perspective_match_func(self, v1, v2, w):
        """
        :param v1: (batch, len, dim)
        :param v2: (batch, len, dim)
        :param w: (perspectives, dim)
        :return: (batch, len, perspectives)
        """
        w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)  # (1,  1,  dim, perspectives)
        v1 = w * torch.stack([v1] * self.perspectives, dim=3)  # (batch, len, dim, perspectives)
        v2 = w * torch.stack([v2] * self.perspectives, dim=3)  # (batch, len, dim, perspectives)
        return functional.cosine_similarity(v1, v2, dim=2)  # (batch, len, perspectives)

    def forward_dense_gcn_layers(self, feat, adj):

        feat_in = feat
        for i in range(1, self.gcn_numbers + 1):
            feat_out = functional.relu(getattr(self, 'gc{}'.format(i))(x=feat_in, adj=adj, mask=None, add_loop=False),
                                       inplace=True)
            feat_out = functional.dropout(feat_out, p=self.dropout, training=self.training)
            feat_in = feat_out
        return feat_out



    def forward(self, data):
        feature_p_init = data["features_1"]
        adj_p = data["edge_index_1"]
        feature_h_init = data["features_2"]
        adj_h = data["edge_index_2"]
        feature_p = self.forward_dense_gcn_layers(feat=feature_p_init, adj=adj_p)  # (batch, len_p, dim)
        feature_h = self.forward_dense_gcn_layers(feat=feature_h_init, adj=adj_h)  # (batch, len_h, dim)

        # ---------- Node-Graph Matching Layer ----------
        attention = self.cosine_attention(feature_p, feature_h)  # (batch, len_p, len_h)

        attention_h = feature_h.unsqueeze(1) * attention.unsqueeze(
            3)  # (batch, 1, len_h, dim) * (batch, len_p, len_h, dim) => (batch, len_p, len_h, dim)
        attention_p = feature_p.unsqueeze(2) * attention.unsqueeze(
            3)  # (batch, len_p, 1, dim) * (batch, len_p, len_h, dim) => (batch, len_p, len_h, dim)

        att_mean_h = self.div_with_small_value(attention_h.sum(dim=2),
                                               attention.sum(dim=2, keepdim=True))  # (batch, len_p, dim)
        att_mean_p = self.div_with_small_value(attention_p.sum(dim=1), attention.sum(dim=1, keepdim=True).permute(0, 2,
                                                                                                                  1))  # (batch, len_h, dim)

        if self.args.match.lower() == "node-graph":
            multi_p = self.multi_perspective_match_func(v1=feature_p, v2=att_mean_h, w=self.mp_w)
            multi_h = self.multi_perspective_match_func(v1=feature_h, v2=att_mean_p, w=self.mp_w)
        else:
            raise NotImplementedError

        match_p = multi_p
        match_h = multi_h

        # ---------- Aggregation Layer ----------
        if self.args.match_agg.lower() == 'bilstm':
            p_agg_bilstm_h0 = torch.zeros(2 * 1, match_p.size(0), self.gcn_last_filter, dtype=torch.float32).to(
                self.device)
            p_agg_bilstm_c0 = torch.zeros(2 * 1, match_p.size(0), self.gcn_last_filter, dtype=torch.float32).to(
                self.device)

            h_agg_bilstm_h0 = torch.zeros(2 * 1, match_h.size(0), self.gcn_last_filter, dtype=torch.float32).to(
                self.device)
            h_agg_bilstm_c0 = torch.zeros(2 * 1, match_h.size(0), self.gcn_last_filter, dtype=torch.float32).to(
                self.device)

            _, (agg_p_last, _) = self.agg_bilstm(match_p, (
            p_agg_bilstm_h0, p_agg_bilstm_c0))  # (batch, seq_len, l) -> (2, batch, hidden_size)
            agg_p = agg_p_last.permute(1, 0, 2).contiguous().view(-1, self.hidden_size * 2)
            _, (agg_h_last, _) = self.agg_bilstm(match_h, (h_agg_bilstm_h0, h_agg_bilstm_c0))
            agg_h = agg_h_last.permute(1, 0, 2).contiguous().view(-1, self.hidden_size * 2)

        elif self.args.match_agg.lower() == 'avg':
            agg_p = torch.mean(match_p, dim=1)
            agg_h = torch.mean(match_h, dim=1)
        elif self.args.match_agg.lower() == 'fc_avg':
            agg_p = torch.mean(self.fc_agg(match_p), dim=1)
            agg_h = torch.mean(self.fc_agg(match_h), dim=1)
        elif self.args.match_agg.lower() == 'max':
            agg_p = torch.max(match_p, dim=1)[0]
            agg_h = torch.max(match_h, dim=1)[0]
        elif self.args.match_agg.lower() == 'fc_max':
            agg_p = torch.max(self.fc_agg(match_p), dim=1)[0]
            agg_h = torch.max(self.fc_agg(match_h), dim=1)[0]
        else:
            raise NotImplementedError

        # option: global aggregation
        if self.global_flag is True:
            global_gcn_agg_p = self.global_aggregation_info(v=feature_p, agg_func_name=self.global_agg)
            global_gcn_agg_h = self.global_aggregation_info(v=feature_h, agg_func_name=self.global_agg)

            agg_p = torch.cat([agg_p, global_gcn_agg_p], dim=1)
            agg_h = torch.cat([agg_h, global_gcn_agg_h], dim=1)

        # ---------- Prediction Layer ----------
        if self.args.task.lower() == 'regression':
            x = torch.cat([agg_p, agg_h], dim=1)
            x = functional.dropout(x, p=self.dropout, training=self.training)
            x = functional.relu(self.predict_fc1(x))
            x = functional.dropout(x, p=self.dropout, training=self.training)
            x = functional.relu(self.predict_fc2(x))
            x = functional.dropout(x, p=self.dropout, training=self.training)
            x = functional.relu(self.predict_fc3(x))
            x = functional.dropout(x, p=self.dropout, training=self.training)
            x = self.predict_fc4(x)
            x = torch.sigmoid(x).squeeze(-1)
            return x
        elif self.args.task.lower() == 'classification':
            sim = functional.cosine_similarity(agg_p, agg_h, dim=1).clamp(min=-1, max=1)
            return sim
        else:
            raise NotImplementedError

