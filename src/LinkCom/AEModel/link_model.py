# !/usr/bin/env python
# -*-coding:utf-8 -*-
# File       : link_model.py
# Author     ：Clark Wang
# version    ：python 3.x
import torch
import torch.nn as nn
from torch.nn import functional as F


class TenorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.tensor_neurons = 32
        self.final_filter = 1024
        super(TenorNetworkModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()
        self.device = self.args.device


    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(1024,
                                                             1024,
                                                             32))

        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(32,
                                                                   2*1024))
        self.bias = torch.nn.Parameter(torch.Tensor(32, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        scoring = torch.mm(torch.t(embedding_1), self.weight_matrix.view(1024, -1))
        scoring = scoring.view(1024, 32)
        scoring = torch.mm(torch.t(scoring), embedding_2)
        combined_representation = torch.cat((embedding_1, embedding_2))
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        return scores


class AEModel(nn.Module):
    def __init__(self, args):
        super(AEModel, self).__init__()
        self.args = args
        self.model = nn.Sequential(
            nn.Linear(in_features=500, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1)
        )
        self.score = TenorNetworkModule(self.args)
        self.fc_1 = nn.Linear(32, 16)
        self.fc_2 = nn.Linear(16, 1)
        # self.fully_connected_first = torch.nn.Linear(self.feature_count,
        #                                              self.args.mlp_neurons)


    def single_ae(self, x):
        return self.model(x)

    def forward(self, data):
        abs_1 = self.single_ae(data["features_1"])
        abs_2 = self.single_ae(data["features_2"])
        abs_1 = abs_1.view(1024, -1)
        abs_2 = abs_2.view(1024, -1)
        # print(abs_1.size())
        # abs_1 = torch.flatten(abs_1)
        # print(abs_1.size())
        # abs_2 = torch.flatten(abs_2)
        scores = self.score(abs_1, abs_2)
        scores = torch.t(scores)
        x = self.fc_1(scores)
        x = self.fc_2(x)
        return x
