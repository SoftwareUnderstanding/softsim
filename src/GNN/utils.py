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