"""
Main shape of data: [batch_size, time(most likely 2), color_channels(3), height, width]
Time will be 2 for  o p t i c a l   f l o w
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

#Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='N/A')
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

device = torch.device('cuda:0' if (torch.cuda.is_available() and args.cuda) else 'cpu')

file = np.load("data/data_all.npz")
"""
x_train = file['X_train']
x_train = torch.tensor(x_train, dtype=torch.float32)
x_train /= 255

y_train = file['y_train']
y_train = torch.tensor(y_train, dtype=torch.float32)

train_data = data_utils.TensorDataset(x_train, y_train)
del x_train
del y_train
trainLoader = data_utils.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
del train_data
"""
print("Train Data Loaded")

x_test = file['X_test']
x_test = torch.tensor(x_test, dtype=torch.float32)
x_test /= 255

y_test = file['y_test']
y_test = torch.tensor(y_test, dtype=torch.float32)

test_data = data_utils.TensorDataset(x_test, y_test)
del x_test
del y_test
trainLoader = data_utils.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
del test_data

print("Test Data Loaded")

from model import Recurrent
net = Recurrent(device)
net.to(device)

loss_function = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    net.train()
    for batch_idx, batch in enumerate(trainLoader):
        X, y_true = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        out = net(X)

        loss = loss_function(out, y_true)
        loss.backward()
        optimizer.step()

        print("Epoch: {} - Training: {}/{} - Loss: {}".format(
            epoch,
            batch_idx * len(X),
            len(trainLoader.dataset),
            loss.item()
        ), end='\r')
    print("")

torch.save(net, 'models/model.pt')
