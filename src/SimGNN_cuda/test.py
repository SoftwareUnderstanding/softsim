# !/usr/bin/env python
# -*-coding:utf-8 -*-
# File       : test.py
# Author     ：Clark Wang
# version    ：python 3.x
import argparse
import os
import logging
import torch

from SimGNN_cuda.simgnn import *
from SimGNN_cuda.utils import *

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
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
                        default="D:\\Projects\\UPM\\GNN\\ran_training.csv",
                        help="DataFrame contains pairs and Sim Score.")

    parser.add_argument("--save-path",
                        type=str,
                        default='D:\\Projects\\UPM\\GNN\\model_folder\\',
                        help="Where to save the trained model")

    parser.add_argument("--load-path",
                        type=str,
                        default=None,
                        help="Load a pretrained model")

    parser.add_argument("--sim_type",
                        type=str,
                        default='sbert_1000',
                        help="Where to save the trained model")

    parser.set_defaults(histogram=True)

    parser.set_defaults(dropout_flag=True)

    parser.add_argument("--epochs",
                        type=int,
                        default=5,
                        help="Number of training epochs. Default is 5.")

    parser.add_argument("--filters-1",
                        type=int,
                        default=512,
                        help="Filters (neurons) in 1st convolution. Default is 128.")

    parser.add_argument("--filters-2",
                        type=int,
                        default=256,
                        help="Filters (neurons) in 2nd convolution. Default is 64.")

    parser.add_argument("--filters-3",
                        type=int,
                        default=128,
                        help="Filters (neurons) in 2nd convolution. Default is 64.")

    parser.add_argument("--tensor-neurons",
                        type=int,
                        default=16,
                        help="Neurons in tensor network layer. Default is 16.")

    parser.add_argument("--bottle-neck-neurons",
                        type=int,
                        default=16,
                        help="Bottle neck layer neurons. Default is 16.")

    parser.add_argument("--batch-size",
                        type=int,
                        default=256,
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
                        default=0.00005,
                        help="Learning rate. Default is 0.002.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5 * (10 ** -4),
                        help="Adam weight decay. Default is 5*10^-4.")

    parser.add_argument("--histogram",
                        dest="histogram",
                        action="store_true")

    parser.add_argument("--device",
                        default=torch.device('cuda' if torch.cuda.is_available else 'cpu'),
                        help="Use Cuda or not")

    parser.add_argument("--func",
                        type=str,
                        default='1000',
                        help="None Linear Function")

    parser.add_argument("--patience",
                        type=int,
                        default=50,
                        help="Patience counter for over-fitting, default as 20")

    return parser.parse_args()

logging.basicConfig(filename="log.txt")
args = parameter_parser()
tab_printer(args)
print(args.histogram)
device = args.device
# data = torch.rand(3, 5).to(device)
# print(data.device)

trainer = SimGNNTrainer(args)
if args.load_path:
    trainer.load()
else:
    trainer.fit()
trainer.score()
if args.save_path:
    trainer.save(args.save_path + 'final.pt')


# a = torch.Tensor([[3.3052e+04, 2.2365e+04, np.nan]])
# if torch.any(torch.isnan(a)):
#     print('*')
#     print(torch.any(torch.isnan(a)))
#     print(torch.where(torch.isnan(a), torch.full_like(a, 0), a))

# args = parameter_parser()
#
# device = args.device
# result_path = 'D:\\Projects\\UPM\\GNN\\model_folder\\final.pt'
# args.load_path = result_path
# trainer = SimGNNTrainer(args)
# trainer.load()
# trainer.score()



