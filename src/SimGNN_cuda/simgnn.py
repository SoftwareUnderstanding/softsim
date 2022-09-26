# !/usr/bin/env python
# -*-coding:utf-8 -*-
# File       : simgnn.py
# Author     ：Clark Wang
# version    ：python 3.x
import glob
import torch
import random
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from torch.nn import functional
from torch_geometric.nn import GCNConv
from layers import AttentionModule, TenorNetworkModule
from utils import process_pair, calculate_loss, format_graph, load_json, load_feature
from sklearn.utils import shuffle

class SimGNN(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SimGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()
        self.device = self.args.device

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram == True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist/torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=True)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=True)

        features = self.convolution_3(features, edge_index)
        return features

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        if self.args.histogram == True:
            hist = self.calculate_histogram(abstract_features_1,
                                            torch.t(abstract_features_2))

        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        if self.args.histogram == True:
            scores = torch.cat((scores, hist), dim=1).view(1, -1)

        scores = torch.nn.functional.normalize(self.fully_connected_first(scores))
        score = torch.nn.functional.relu(self.scoring_layer(scores))
        return score

# device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
class SimGNNTrainer(object):
    def __init__(self, args):
        self.args = args
        self.embedding_len = 1024
        self.get_pairs()
        self.setup_model()
        # self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    def setup_model(self):
        self.model = SimGNN(self.args, self.embedding_len).to(self.args.device)

    def get_pairs(self):
        # data = glob.glob(self.args.data_path + '*.pt')
        data = pd.read_csv(self.args.score_path)
        ### Pairs
        self.testing_pairs= data.sample(frac=0.4)
        self.training_pairs = data[~data.index.isin(self.testing_pairs.index)]
        self.testing_pairs = shuffle(self.testing_pairs)
        self.training_pairs = shuffle(self.training_pairs)
        # print(self.training_pairs.head())


    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: List of lists with batches.
        """
        # random.shuffle(self.training_pairs)
        batches = []
        for graph in range(0, len(self.training_pairs), self.args.batch_size):
            batches.append(self.training_pairs[graph:graph+self.args.batch_size])
        return batches

    ### need to train the datatype
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
            # print(prediction)
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
            patience = 10
            trigger_times = 0
            batches = self.create_batches()
            self.loss_sum = 0
            main_index = 0
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
                loss_score = self.process_batch(batch)
                main_index = main_index + len(batch)
                self.loss_sum = self.loss_sum + loss_score * len(batch)
                loss = self.loss_sum / main_index
                self.training_loss.append(loss)
                epochs.set_description(f"Epoch:{epoch} Batch:{index} (Loss=%g)" % round(loss, 5))
                if loss > last_loss:
                    trigger_times += 1

                    print('Trigger Times:', trigger_times)
                    last_loss = loss
                    if trigger_times >= patience:
                        print(f"Oh, Stopped at {epoch} epoches, {index} batches, so sad!")
                        break
                else:
                    last_loss = loss
            if self.args.save_path:
                self.save(self.args.save_path + f'epoch_{epoch}.pt')
        file = open(self.args.save_path + 'training.txt', 'w')
        for i in self.training_loss:
            s = str(i) + '\n'
            file.write(s)
        file.close()




    def score(self):
        print("\n\nModel evaluation.\n")
        self.model.eval()
        self.scores = []
        self.ground_truth = []
        for _, row in self.testing_pairs.iterrows():
            data = self.transfer_to_torch(row)
            self.ground_truth.append(data['target'].item())
            prediction = self.model(data).item()
            # print(prediction)
            self.scores.append(calculate_loss(prediction, data['target'].item()))
        self.print_evaluation()

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        # print(self.ground_truth)
        # print(type(self.ground_truth))
        norm_ged_mean = np.mean(self.ground_truth)
        base_error = np.mean([(n - norm_ged_mean) ** 2 for n in self.ground_truth])
        model_error = np.mean(self.scores)
        print("\nBaseline error: " + str(round(base_error, 5)) + ".")
        print("\nModel test error: " + str(round(model_error, 5)) + ".")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load(self.args.load_path))