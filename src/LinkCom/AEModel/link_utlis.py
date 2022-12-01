# !/usr/bin/env python
# -*-coding:utf-8 -*-
# File       : link_utlis.py
# Author     ：Clark Wang
# version    ：python 3.x
import json
import torch



def fusion_matrix(data, max_row = 9176):
    temp = list()
    if len(data) == max_row:
        for k, v in data.items():
            temp_v = v.reshape(1024, 1)
            temp.append(temp_v)
        temp = tuple(temp)
        result = torch.cat(temp, -1)
    else:
        for k, v in data.items():
            temp_v = v.reshape(1024, 1)
            temp.append(temp_v)
        temp = tuple(temp)
        temp_result = torch.cat(temp, -1)
        zero_lines = max_row - len(data)
        zero_matrix = torch.zeros(1024, zero_lines)
        result = torch.cat((temp_result, zero_matrix), -1)
    return result


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


def link_nodes(data_dir):
    f = open(data_dir)
    data = json.load(f)
    res = []
    for keys, vals in data.items():
        if vals[0][0] == 'None':
            pass
        else:
            res.append(keys)
            for to_node in vals[0]:
                res.append(to_node)
    return list(set(res))


def fusion(data, link, row):
    temp = list()
    for k, v in data.items():
        if k in link:
            temp_v = v.reshape(1024, 1)
            temp.append(temp_v)
    temp = tuple(temp)
    if len(temp) == row:
        result = torch.cat(temp, -1)
    else:
        temp_result = torch.cat(temp, -1)
        zero_lines = row - len(temp)
        zero_matrix = torch.zeros(1024, zero_lines)
        result = torch.cat((temp_result, zero_matrix), -1)
    return result


def fusion_min(data, link, row):
    temp = list()
    for k, v in data.items():
        if k in link:
            temp_v = v.reshape(1024, 1)
            temp.append(temp_v)
            if len(temp) == row:
                break

    temp = tuple(temp)
    result = torch.cat(temp, -1)
    return result


def fusion_avg(data, link, row = 500):
    temp = list()
    i = 0
    for k, v  in data.items():
        if k in link:
            temp_v = v.reshape(1024, 1)
            temp.append(temp_v)
            i += 1
        if i == row:
            break
    if len(temp) < row:
        temp_result = torch.cat(temp, -1)
        zero_lines = row - len(temp)
        zero_matrix = torch.zeros(1024, zero_lines)
        result = torch.cat((temp_result, zero_matrix), -1)
    else:
        result = torch.cat(temp, -1)
    return result


def calculate_ranking_correlation(rank_corr_function, prediction, target):
    """
    Calculating specific ranking correlation for predicted values.
    :param rank_corr_function: Ranking correlation function.
    :param prediction: Vector of predicted values.
    :param target: Vector of ground-truth values.
    :return ranking: Ranking correlation value.
    """
    temp = prediction
    r_prediction = np.empty_like(temp)
    r_prediction[temp] = np.arange(len(prediction))

    temp = target
    r_target = np.empty_like(temp)
    r_target[temp] = np.arange(len(target))

    return rank_corr_function(r_prediction, r_target).correlation


def calculate_prec_at_k(k, prediction, target):
    """
    Calculating precision at k.
    """

    # increase k in case same similarity score values of k-th, (k+i)-th elements
    target_increase = np.sort(target)[::-1]
    target_value_sel = (target_increase >= target_increase[k - 1]).sum()
    target_k = max(k, target_value_sel)

    best_k_pred = prediction.argsort()[::-1][:k]
    best_k_target = target.argsort()[::-1][:target_k]

    return len(set(best_k_pred).intersection(set(best_k_target))) / k


def precision(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(k)
    return result