# !/usr/bin/env python
# -*-coding:utf-8 -*-
# File       : modelApply.py
# Author     ：Clark Wang
# version    ：python 3.x
from simgnn import *
import os


model = SimGNN()

temp_list = os.listdir('D:\\Projects\\UPM\\GNN\\results\\')
model_para = []
for i in temp_list:
    print(i[-2:])


