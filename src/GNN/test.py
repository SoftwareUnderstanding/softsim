# !/usr/bin/env python
# -*-coding:utf-8 -*-
# File       : test.py
# Author     ：Clark Wang
# version    ：python 3.x
import argparse
import torch
from base_model import *

def parameter_parser():
    parser = argparse.ArgumentParser(description="Run SimGNN.")

    parser.add_argument("--data-path",
                        nargs="?",
                        default="D:\\SoftwareSim\\post_process\\",
                        help="Json data path for linking.")

    parser.add_argument("--json-path",
                        nargs="?",
                        default="D:\\SoftwareSim\\final_data\\",
                        help="Folder with graph pair pts.")

    parser.add_argument("--score-path",
                        nargs="?",
                        default="D:\\Projects\\UPM\\GNN\\data_process\\training.csv",
                        help="DataFrame contains pairs and Sim Score.")

    parser.add_argument("--save-path",
                        type=str,
                        default='D:\\Projects\\UPM\\GNN\\model_folder\\',
                        help="Where to save the trained model")

    parser.add_argument("--load-path",
                        type=str,
                        default='D:\\Projects\\UPM\\GNN\\SimGNN_cuda\\unfin\\epoch_5.pt',
                        help="Load a pretrained model")

    parser.add_argument("--sim_type",
                        type=str,
                        default='sbert_100',
                        help="Where to save the trained model")

    parser.set_defaults(histogram=True)

    parser.set_defaults(dropout_flag=True)

    parser.add_argument("--epochs",
                        type=int,
                        default=5,
                        help="Number of training epochs. Default is 5.")

    parser.add_argument("--tensor-neurons",
                        type=int,
                        default=16,
                        help="Neurons in tensor network layer. Default is 16.")

    parser.add_argument("--batch-size",
                        type=int,
                        default=512,
                        help="Number of graph pairs per batch. Default is 512.")

    parser.add_argument("--bins",
                        type=int,
                        default=16,
                        help="Similarity score bins. Default is 16.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
                        help="Dropout probability. Default is 0.5.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=5 * (10 ** -6),
                        help="Learning rate. Default is 0.002.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5 * (10 ** -4),
                        help="Adam weight decay. Default is 5*10^-4.")

    parser.add_argument("--feature_length",
                            type=int,
                            default=1024,
                            help="Filters (neurons) in 1st convolution. Default is 128.")

    parser.add_argument("--mlp_neurons",
                        type=int,
                        default=64,
                        help="Filters (neurons) in 1st convolution. Default is 128.")

    parser.add_argument("--device",
                            default=torch.device('cuda' if torch.cuda.is_available else 'cpu'),
                            help="Use Cuda or not")

    parser.add_argument("--filters",
                            type=str,
                            default='512',
                            help="Where to save the trained model")

    parser.add_argument("--conv",
                        type=str,
                        default='gcn',
                        help="Where to save the trained model")

    parser.add_argument("--histogram",
                        dest="histogram",
                        action="store_true")

    parser.add_argument("--patience",
                        type=int,
                        default=200,
                        help="Patience counter for over-fitting, default as 20")

    return parser.parse_args()


a = parameter_parser()
trainer = BaseTrainer(a)
trainer.fit()