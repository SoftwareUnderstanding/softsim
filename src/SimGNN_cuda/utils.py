# !/usr/bin/env python
# -*-coding:utf-8 -*-
# File       : utils.py
# Author     ：Clark Wang
# version    ：python 3.x
import math

import numpy as np
from texttable import Texttable
import torch
import json


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def process_pair(path):
    """
    Reading a json file with a pair of graphs.
    :param path: Path to a JSON file.
    :return data: Dictionary with data.
    """
    data = torch.load(path)
    return data

def dis_sim(data):
    data = 2**data

def calculate_loss(prediction, target):
    """
    Calculating the squared loss on the normalized GED.
    :param prediction: Predicted log value of GED.
    :param target: Factual log transofmed GED.
    :return score: Squared error.
    """
    # prediction = -math.log(prediction)
    # target = -math.log(target)
    score = (prediction-target)**2
    return score

def calculate_normalized_ged(data):
    """
    Calculating the normalized GED for a pair of graphs.
    :param data: Data table.
    :return norm_ged: Normalized GED score.
    """
    norm_ged = data["ged"]/(0.5*(len(data["labels_1"])+len(data["labels_2"])))
    return norm_ged


def load_json(path):
    data = json.load(open(path))
    return data

def load_feature(data):
    output = []
    for keys, vals in data.items():
        output.append(vals.view(1, -1))
    return torch.cat(output, dim=0)


def format_graph(data):
    node_index = list(data.keys())
    from_list, to_list = [], []
    for keys, vals in data.items():
        if vals[0][0] == 'None':
            pass
        else:
            from_node_index = node_index.index(keys)
            for to_node in vals[0]:
                to_node_index = node_index.index(to_node)
                from_list.append(from_node_index)
                to_list.append(to_node_index)
    return [from_list, to_list]

# def normal_dist(x, sd=0.001):
#     prob_density = (np.pi*sd) * np.exp(-0.5*((x)/sd)**2)
#     return prob_density


def none_linear_func(lin, data):
    funcs = {
        # "cube": lambda x: (x*10)**3,
        "100": lambda x: (1/(x))**2 * (-math.log(x)) if x != 0 else 0,
        "1000": lambda x: (1/(x))**3 * (-math.log(x)) if x != 0 else 0,
        "10000": lambda x: (1/(x))**4 * (-math.log(x)) if x != 0 else 0,
        "exp": lambda x: math.exp(math.tanh(x)*2),
        "tanh": lambda x: 1/(1 - math.tanh(x+0.5)),
        "none": lambda x: x
        # "norm" : lambda x: normal_dist(x) if x>=0 else -normal_dist(x)
        # "con": lambda x: -math.log(x) * 10 if x > 0 else (x * 100)**2
    }
    return np.float64(funcs[lin](data))
a = none_linear_func('100', 0.1)
b = none_linear_func('100', 0.9)
print(a)
print(b)
print(type(a))



# data = process_pair('D:\\Projects\\UPM\\GNN\\data\\final_ae_Data\\Zasder3_Latent-Neural-Differential-Equations-for-Video-Generation.pt')
# output = load_feature(data)
# print(output)
# print(torch.cat(output, dim=0).size())
# data = load_json('D:\\Projects\\UPM\\GNN\\data\\final_data\\2-Chae_A-NDFT.json')
# a = format_graph(data)
# print(torch.FloatTensor(a))

