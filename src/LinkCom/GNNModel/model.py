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
from torch.nn import functional
from torch import nn
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, SuperGATConv
from sklearn.utils import shuffle
from layers import *
from utlis import *
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, kendalltau


class BaseModel(torch.nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.in_channels = args.feature_length
        self.device = args.device
        filters = self.args.filters.split('_')
        self.gcn_filters = [int(n_filter) for n_filter in filters]
        self.gcn_numbers = len(self.gcn_filters)
        self.gcn_last_filter = self.gcn_filters[-1]
        self.args.final_filter = self.gcn_last_filter
        gcn_parameters = [dict(in_channels=self.gcn_filters[i - 1], out_channels=self.gcn_filters[i]) for i
                          in range(1, self.gcn_numbers)]
        gcn_parameters.insert(0, dict(in_channels=self.in_channels, out_channels=self.gcn_filters[0]))

        self.conv_layer_dict = {
            "gcn": dict(constructor=GCNConv, kwargs=gcn_parameters),
            "gat": dict(constructor=GATv2Conv, kwargs=gcn_parameters),
            "supergat": dict(constructor=SuperGATConv, kwargs=gcn_parameters),
            "sage": dict(constructor=SAGEConv, kwargs=gcn_parameters)
        }
        print(gcn_parameters)
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram == True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons


    def setup_layers(self):
        self.calculate_bottleneck_features()
        conv = self.conv_layer_dict[self.args.conv]
        constructor = conv['constructor']
        # print(constructor)
        setattr(self, 'gc{}'.format(1), constructor(**conv['kwargs'][0]))
        for i in range(1, self.gcn_numbers):
            setattr(self, 'gc{}'.format(i + 1), constructor(**conv['kwargs'][i]))

        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)

        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.mlp_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.mlp_neurons, 1)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        if torch.any(torch.isnan(scores)):
            # print(scores)
            scores = torch.where(torch.isnan(scores), torch.full_like(scores, 0), scores)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist/torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def convolutional_pass(self, adj, feat_in):
        for i in range(1, self.gcn_numbers + 1):
            feat_out = functional.relu(getattr(self, 'gc{}'.format(i))(feat_in, adj),
                                       inplace=True)
            feat_out = functional.dropout(feat_out, p=self.args.dropout, training=True)
            feat_in = feat_out
        return feat_out

    def forward(self, data):
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)

        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        # print(abstract_features_1.size())
        # print(abstract_features_2.size())
        # print(pooled_features_1.size())
        # print(pooled_features_2.size())

        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)
        if self.args.histogram == True:
            hist = self.calculate_histogram(abstract_features_1,
                                            torch.t(abstract_features_2))

            scores = torch.cat((scores, hist), dim=1).view(1, -1)

        scores = torch.nn.functional.normalize(self.fully_connected_first(scores))
        score = torch.nn.functional.relu(self.scoring_layer(scores))
        return score


class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.embedding_len = args.feature_length
        self.get_pairs()
        self.setup_model()

    def setup_model(self):

        self.model = BaseModel(self.args)
        self.model = nn.DataParallel(self.model)
        self.model.to(self.args.device)

    def get_pairs(self):
        # data = glob.glob(self.args.data_path + '*.pt')
        data = pd.read_csv(self.args.score_path)
        # data = data.sample(frac = 0.1, random_state=42)
        self.training_pairs, self.testing_pairs = train_test_split(data, test_size=0.2, random_state=42)
        self.training_pairs, self.validation_pairs = train_test_split(self.training_pairs, test_size=0.2, random_state=42)

    def create_batches(self, data):
        """
        Creating batches from the training graph list.
        :return batches: List of lists with batches.
        """
        # random.shuffle(self.training_pairs)
        batches = []
        for graph in range(0, len(data), self.args.batch_size):
            batches.append(data[graph:graph+self.args.batch_size])
        return batches

    def transfer_to_torch(self, data):
        '''
        :param data: data.series from Score.csv
        :return: graph pair as dict
        '''
        new_dict = {}
        graph_1 = process_pair(self.args.data_path + data['graph_1'] + '.pt')
        graph_2 = process_pair(self.args.data_path + data['graph_2'] + '.pt')
        json_g_1 = load_json(self.args.json_path + data['graph_1'] + '.json')
        json_g_2 = load_json(self.args.json_path + data['graph_2'] + '.json')
        # new_dict['graph_1'], new_dict['graph_2'] = graph_1, graph_2
        new_dict['features_1'] = load_feature(graph_1).to(self.args.device)
        new_dict['features_2'] = load_feature(graph_2).to(self.args.device)
        new_dict['target'] = torch.from_numpy(np.float64(data[self.args.sim_type]).reshape(1, 1)).view(-1).float().to(self.args.device)
        # new_dict['target'] = torch.from_numpy(none_linear_func(self.args.func, data[self.args.sim_type]).reshape(1, 1)).view(-1).float().to(self.args.device)
        # new_dict['target'] = data[self.args.sim_type]
        edge_1 = torch.LongTensor(format_graph(json_g_1)).to(self.args.device)
        edge_2 = torch.LongTensor(format_graph(json_g_2)).to(self.args.device)
        new_dict['edge_index_1'], new_dict['edge_index_2'] = edge_1, edge_2
        return new_dict

    def process_batch(self, batch):
        self.optimizer.zero_grad()
        losses = 0
        for _, graph_pairs in batch.iterrows():
            data = self.transfer_to_torch(graph_pairs)
            target = data['target']
            # data = data.to(self.device)
            prediction = self.model(data).view(1)
            # prediction = torch.from_numpy(np.float64(none_linear_func(self.args.func, prediction)).reshape(1, 1))
            # print(type(prediction))
            # print(target)
            losses = losses + torch.nn.functional.mse_loss(target, prediction)
        losses.backward(retain_graph=True)
        self.optimizer.step()
        loss = losses.item()
        return loss

    def fit(self):
        self.training_loss = []

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        self.model.train()
        epochs = trange(self.args.epochs, leave=True, desc="Epoch")

        for epoch in epochs:
            last_loss = float('inf')
            batches = self.create_batches(self.training_pairs)
            self.loss_sum = 0
            main_index = 0
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
                loss_score = self.process_batch(batch)
                main_index = main_index + len(batch)
                self.loss_sum = self.loss_sum + loss_score * len(batch)
                loss = self.loss_sum / main_index
                loss = loss_score / self.args.batch_size
                self.training_loss.append(loss)
                epochs.set_description(f"Epoch:{epoch} Batch:{index} (Loss=%g)" % round(loss, 5))
            self.val_score = []
            for _, row in self.validation_pairs.iterrows():
                val_data = self.transfer_to_torch(row)
                prediction = self.model(val_data).item()
                val_curr_score = calculate_loss(prediction, val_data['target'].item())
                self.val_score.append(val_curr_score)
            training_met = np.mean(self.training_loss)
            val_met = np.mean(self.val_score)
            print(f"training_loss: {training_met}, val_loss: {val_met}")

            if (epoch + 1) % 20 == 0:
                self.score()
                if self.args.save_path:
                    self.save(self.args.save_path + f'epoch_{epoch}.pt')

    def score(self):
        print("\n\nModel evaluation.\n")
        self.model.eval()
        self.scores = []
        self.ground_truth = []

        self.rho_list = []
        self.tau_list = []
        self.prec_at_10_list = []
        self.prec_at_20_list = []
        file_list = ["scores", "rho_list", "tau_list", "prec_at_10_list", "prec_at_20_list"]
        data_list = [self.scores, self.rho_list, self.tau_list, self.prec_at_10_list, self.prec_at_20_list]
        batches = self.create_batches(self.testing_pairs)
        for index, batch in enumerate(batches):
            self.batch_scores = []
            self.batch_ground_truth = []
            self.batch_prediction = []
            for _, row in batch.iterrows():
                data = self.transfer_to_torch(row)
                self.batch_ground_truth.append(data['target'].item())
                self.ground_truth.append(data['target'].item())
                prediction = self.model(data).item()
                self.batch_prediction.append(prediction)
                # prediction_mat = np.float64(prediction).item()
                self.batch_scores.append(calculate_loss(prediction, data['target'].item()))
            self.scores.append(np.mean(self.batch_scores))
            self.rho_list.append(spearmanr(self.batch_prediction, self.batch_ground_truth).correlation)
            self.tau_list.append(kendalltau(self.batch_prediction, self.batch_ground_truth).correlation)
            self.prec_at_10_list.append(precision(self.batch_prediction, self.batch_ground_truth, 10))
            self.prec_at_20_list.append(precision(self.batch_prediction, self.batch_ground_truth, 20))
        for file_index in range(len(file_list)):
            data_index = data_list[file_index]
            with open(self.args.save_path + f'{file_list[file_index]}.txt', 'w') as f:
                for line in data_index:
                    f.write(f"{line}\n")
        self.print_evaluation()

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        norm_ged_mean = np.mean(self.ground_truth)
        base_error = np.mean([(n - norm_ged_mean) ** 2 for n in self.ground_truth])
        model_error = np.mean(self.scores)
        model_rho = np.mean(self.rho_list)
        model_tau = np.mean(self.tau_list)
        model_10 = np.mean(self.prec_at_10_list)
        model_20 = np.mean(self.prec_at_20_list)
        print("\nBaseline error: " + str(round(base_error, 5)) + ".")
        print("\nModel test error: " + str(round(model_error, 5)) + ".")
        print("\nSpearman's rho: " + str(round(model_rho, 5)) + ".")
        print("\nKendall's tau: " + str(round(model_tau, 5)) + ".")
        print("\np@10: " + str(round(model_10, 5)) + ".")
        print("\np@20: " + str(round(model_20, 5)) + ".")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load(self.args.load_path))

