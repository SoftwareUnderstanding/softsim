# !/usr/bin/env python
# -*-coding:utf-8 -*-
# File       : link_trainer.py
# Author     ：Clark Wang
# version    ：python 3.x
from link_utlis import *
from link_model import *
import torch
from torch.nn import functional as F
from torch import nn
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm, trange
import numpy as np
from scipy.stats import spearmanr, kendalltau



class trainer(object):
    def __init__(self, args):
        self.args = args
        self.get_pairs()
        self.setup_model()

    def get_pairs(self):
        data = pd.read_csv(self.args.score_path)
        data = data.sample(frac = 0.2, random_state=42)
        self.training_pairs, self.testing_pairs = train_test_split(data, test_size=0.2, random_state=42)
        self.training_pairs, self.validation_pairs = train_test_split(self.training_pairs, test_size=0.2,
                                                                      random_state=42)

    def setup_model(self):

        self.model = AEModel(self.args)
        self.model = nn.DataParallel(self.model)
        self.model.to(self.args.device)

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

    def transfer_to_torch(self,data):
        new_dict = {}
        pt_1, pt_2 = (self.args.data_path + data['graph_1'] + '.pt'), (self.args.data_path + data['graph_2'] + '.pt')
        link_1, link_2 = (self.args.json_path + data['graph_1'] + '.json'), (self.args.json_path + data['graph_2'] + '.json')
        dict_1, dict_2 = torch.load(pt_1), torch.load(pt_2)
        json_1, json_2 = link_nodes(link_1), link_nodes(link_2)
        data_1 = fusion_avg(dict_1, json_1).to(self.args.device)
        data_2 = fusion_avg(dict_2, json_2).to(self.args.device)
        # data_1 = fusion_matrix(torch.load(self.args.data_path + data['graph_1'] + '.pt')).to(self.args.device)
        # data_2 = fusion_matrix(torch.load(self.args.data_path + data['graph_2'] + '.pt')).to(self.args.device)
        new_dict["features_1"] = data_1.unsqueeze(0)
        new_dict["features_2"] = data_2.unsqueeze(0)
        new_dict['target'] = torch.from_numpy(np.float64(data[self.args.sim_type]).reshape(1, 1)).view(-1).float().to(self.args.device)
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