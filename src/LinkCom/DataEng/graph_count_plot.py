import os
import pandas as pd
import numpy as np
import torch
from eng_utlis import *
import matplotlib.pyplot as plt
import math
from collections import Counter

json_path = "D:\\SoftwareSim\\final_data\\"
pt_path = "D:\\SoftwareSim\\post_process\\"
node_count = []
link_count = []
range = list()
i = 0
for name in os.listdir(pt_path):
    name = name.split('.')
    name = name[0] + ".json"
    data = load_json(json_path + name)
    # print(data)
    temp = format_graph(data)
    a, b = temp[0], temp[1]
    abc = Counter(a+b)

    link_count.append(float(math.log10(len(a))))
    # abc = list(set(a).symmetric_difference(set(b)))
    # print(len(abc))
    node_count.append(float(math.log10(len(abc))))
    # node_count.append(len(abc))
    # link_count.append(len(a))
    range.append(i)
    i += 1

node_hist,node_bins=np.histogram(node_count,bins=np.linspace(0,4.8,16))
link_hist,link_bins=np.histogram(link_count,bins=np.linspace(0,4.8,16))
bins = list(link_bins)
bins.pop()
fig = plt.figure()
plt.plot(node_hist, label="nodes", marker="4")
plt.plot(link_hist, label="links", marker='*')
plt.xlabel("#x power to 10")
plt.ylabel("#graph count")
plt.grid(axis='y', linestyle = '--')
leg = plt.legend(loc='upper right')
plt.show()