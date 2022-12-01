# !/usr/bin/env python
# -*-coding:utf-8 -*-
# File       : link_test.py
# Author     ：Clark Wang
# version    ：python 3.x
import argparse
from link_trainer import *

def parameter_parser():
    parser = argparse.ArgumentParser(description="Run AE Based Model.")

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
                        default="D:\\Projects\\Lucia\\link_testing_data.csv",
                        help="DataFrame contains pairs and Sim Score.")

    parser.add_argument("--save-path",
                        type=str,
                        default='D:\\Projects\\\Result\\test\\',
                        help="Where to save the trained model")

    parser.add_argument("--load-path",
                        type=str,
                        default=None,
                        help="Load a pretrained model")

    parser.add_argument("--sim_type",
                        type=str,
                        default='sbert_100',
                        help="Where to save the trained model")


    parser.add_argument("--epochs",
                        type=int,
                        default=100,
                        help="Number of training epochs. Default is 5.")


    parser.add_argument("--batch-size",
                        type=int,
                        default=32,
                        help="Number of graph pairs per batch. Default is 512.")


    parser.add_argument("--learning-rate",
                        type=float,
                        default=1 * (10 ** -4),
                        help="Learning rate. Default is 0.002.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5 * (10 ** -4),
                        help="Adam weight decay. Default is 5*10^-4.")

    parser.add_argument("--feature_length",
                            type=int,
                            default=1024,
                            help="Input Length. Default is 1024.")


    parser.add_argument("--device",
                            default=torch.device('cuda' if torch.cuda.is_available else 'cpu'),
                            help="Use Cuda or not")


    parser.add_argument("--patience",
                        type=int,
                        default=2000,
                        help="Patience counter for over-fitting, default as 20")

    parser.set_defaults(continue_training=True)

    return parser.parse_args()


args = parameter_parser()
device = args.device


trainer = trainer(args)
trainer.fit()
