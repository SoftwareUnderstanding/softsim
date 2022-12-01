# !/usr/bin/env python
# -*-coding:utf-8 -*-
# File       : sim_score_plot.py
# Author     ：Clark Wang
# version    ：python 3.x
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data_source = "D:\\Projects\\UPM\\GNN\\data_process\\training.csv"
data = pd.read_csv(data_source)
minilm = data["miniLM"]
sbert = data["sbert"]
tsdae = data["tsdae"]

mini_hist, mini_bins = np.histogram(minilm, bins=np.linspace(0,1,20))
sbert_hist, _ = np.histogram(sbert, bins=np.linspace(0,1,20))
tsdae_hist, _ = np.histogram(tsdae, bins=np.linspace(0,1,20))
mini_bins = list(mini_bins)
mini_bins.pop()
fig = plt.figure()
plt.plot(mini_bins, mini_hist, label="miniLM", marker="4")
plt.plot(mini_bins, sbert_hist, label="Distil-SBert", marker="*")
plt.plot(mini_bins, tsdae_hist, label="TSDAE", marker="o")
plt.yticks([0, 100000, 200000, 300000, 400000], [r"0", r"100k", r"200k", r'300k', r'400k'])
plt.xlabel("#cosine similarity score")
plt.ylabel("#training pairs")
leg = plt.legend(loc='upper right')
plt.grid(axis='y', linestyle = '--')
plt.show()