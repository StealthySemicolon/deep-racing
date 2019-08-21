"""
Main shape of data: [batch_size, time(most likely 2), color_channels(3), height, width]
Time will be 2 for  o p t i c a l   f l o w
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

#Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

X_train, X_test, y_train, y_test = np.load("data/data_all.npz")
