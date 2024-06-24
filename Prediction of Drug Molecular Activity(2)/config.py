# coding=utf-8
# import argparse
#
# parser = argparse.ArgumentParser(description='Configuration for training the model')
#
# #Add command line parameters
# #These two parameters must be specified
# parser.add_argument('--my_dataset', type=str, required=True, help='Name of the dataset')
# parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file')
#
# parser.add_argument('--dropout', type=float, default=0.22, help='Dropout rate')
# parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
# parser.add_argument('--my_scheduler', type=bool, default=True, help='Use learning rate scheduler or not')
# parser.add_argument('--lr', type=float, default=0.01, help='Learning rate or initial learning rate for cosine annealing')
# parser.add_argument('--epoch', type=int, default=180, help='Number of epochs')
# parser.add_argument('--max_atoms', type=int, default=100, help='Maximum number of atoms in a molecule')
# parser.add_argument('--alpha', type=float, default=0.5, help='The alpha parameter for focal loss')
#
#
# args = parser.parse_args()
#
#
# DROPOUT = args.dropout
# PATIENCE = args.patience
# MY_SCHEDULER = args.my_scheduler
# LR = args.lr
# EPOCH = args.epoch
# MAX_ATOMS = args.max_atoms
# my_dataset = args.my_dataset
# dataset_path = args.dataset_path
# ALPHA = args.alpha
#
#
#
# print('Configuration:')
# print(f'Dropout: {DROPOUT}')
# print(f'Patience: {PATIENCE}')
# print(f'Use Scheduler: {MY_SCHEDULER}')
# print(f'Learning Rate: {LR}')
# print(f'Epochs: {EPOCH}')
# print(f'Max Atoms: {MAX_ATOMS}')
# print(f'Dataset Name: {my_dataset}')
# print(f'Dataset Path: {dataset_path}')
# print(f'Alpha: {ALPHA}')



DROPOUT = 0.25
PATIENCE = 10
MY_SCHEDULER = True
LR = 0.01
EPOCH = 30
MAX_ATOMS = 100
my_dataset = "casp"
dataset_path="./data/casp.xlsx"
ALPHA = 0.5