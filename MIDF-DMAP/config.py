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


#Whether to use pre extracted 3D features
D3_INFO_PATH=None
# D3_INFO_PATH=f"./{my_dataset}_Intermediate/part1.npz"

BATCH_SIZE=32
OPTIM = 'Adam'

#Whether to use CUDA
USE_CUDA=True


#THR dataset
#Preprocessing settings
MAX_ATOMS = 100
my_dataset = "THR"
dataset_path="./data/THR.xlsx"

#Training settings
DROPOUT = 0.25
#Early Stop Method
PATIENCE = 30
MY_SCHEDULER = True
#learning rate
LR = 0.01
#EPOCH must be a multiple of 3 if MY_SCHEDULER = True
EPOCH = 90
#About Focal loss
ALPHA = 0.5


#CASP dataset
# MAX_ATOMS = 100
# my_dataset = "CASP"
# dataset_path= "./data/CASP.xlsx"
#
# DROPOUT = 0.25
# PATIENCE = 10
# MY_SCHEDULER = True
# LR = 0.01
# EPOCH = 90
# ALPHA = 0.5


#CLINTOX dataset
# MAX_ATOMS=140
# my_dataset = "CLINTOX"
# dataset_path= "./data/CLINTOX.xlsx"
#
# DROPOUT = 0.25
# PATIENCE = 20
# MY_SCHEDULER = True
# LR = 0.01
# EPOCH = 90
# ALPHA = 0.08

#SIDER-HD dataset
# MAX_ATOMS=492
# my_dataset = "SIDER"
# dataset_path= "./data/SIDER-HD.xlsx"
#
# DROPOUT = 0.25
# PATIENCE = 20
# MY_SCHEDULER = True
# LR = 0.01
# EPOCH = 90
# ALPHA = 0.55


#HGFR dataset
# MAX_ATOMS=105
# my_dataset = "HGFR"
# dataset_path= "./data/HGFR.xlsx"
#
# DROPOUT = 0.25
# PATIENCE = 20
# MY_SCHEDULER = True
# LR = 0.01
# EPOCH = 90
# ALPHA = 0.88


#ABL dataset
# MAX_ATOMS=105
# my_dataset = "ABL"
# dataset_path= "./data/ABL.xlsx"
#
# #训练设置
# DROPOUT = 0.25
# PATIENCE = 20
# MY_SCHEDULER = True
# LR = 0.01
# EPOCH = 120
# ALPHA = 0.75

#BACE dataset rm split
# MAX_ATOMS=100
# my_dataset = "BACErm"
# dataset_path= "./data/BACE_rm.xlsx"
#
# OPTIM = 'SGD'
# DROPOUT = 0.35
# PATIENCE = 30
# MY_SCHEDULER = True
# LR = 0.015
# EPOCH = 90
# ALPHA = 0.35


#BACE dataset sd split
# MAX_ATOMS=100
# my_dataset = "BACE"
# dataset_path= "./data/BACE_scaffold.xlsx"
#
# OPTIM = 'SGD'
# DROPOUT = 0.35
# PATIENCE = 30
# MY_SCHEDULER = True
# LR = 0.015
# EPOCH = 120
# ALPHA = 0.35


#BBBP dataset random split
# MAX_ATOMS=132
# my_dataset = "BBBPrm"
# dataset_path= "./data/BBBP_rm.xlsx"
#
# DROPOUT = 0.55
# PATIENCE = 60
# MY_SCHEDULER = True
# LR = 0.012
# EPOCH = 270
# ALPHA = 0.75


#BBBP dataset scaffold split
# MAX_ATOMS=132
# my_dataset = "BBBPuseScaffold"
# dataset_path= "./data/BBBP_useScaffold.xlsx"
# DROPOUT = 0.35
# PATIENCE = 35
# MY_SCHEDULER = True
# LR = 0.01
# EPOCH = 120
# ALPHA = 0.5