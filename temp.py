import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import time

file = np.load("data/data_all.npz")

x_train = file['X_train']
x_train = torch.tensor(x_train, dtype=torch.float32)
x_train /= 255