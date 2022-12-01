# !/usr/bin/env python
# -*-coding:utf-8 -*-
# File       : get_sub_set.py
# Author     ：Clark Wang
# version    ：python 3.x
import pandas as pd
import json
import os
import random
import itertools
from collections import Counter

def get_software(file_name = "", count = 500):
    training_source = "D:\\SoftwareSim\\training.csv"
    val_1_source = "D:\\SoftwareSim\\val_1.csv"
    val_2_source = "D:\\SoftwareSim\\val_2.csv"
    val_3_source = "D:\\SoftwareSim\\val_3.csv"
    training = pd.read_csv(training_source)
    val_1 = pd.read_csv(val_1_source)
    val_2 = pd.read_csv(val_2_source)
    val_3 = pd.read_csv(val_3_source)
    temp_list = os.listdir("D:\\SoftwareSim\\post_process")
    target_list = random.sample(temp_list, count)
    names = []
    for i in target_list:
        names.append(i.split('.')[0])
    data = pd.concat([training, val_1, val_2, val_3])
    temp = data[(data['graph_1'].isin(names)) & (data['graph_2'].isin(names))]
    print(len(temp))
    temp.to_csv(f"{file_name}.csv")



get_software('median', 500)