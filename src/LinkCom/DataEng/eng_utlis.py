# !/usr/bin/env python
# -*-coding:utf-8 -*-
# File       : eng_utlis.py
# Author     ：Clark Wang
# version    ：python 3.x


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