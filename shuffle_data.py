import numpy as np
import argparse
import random

parser = argparse.ArgumentParser(description='Data Shuffler')
parser.add_argument('--start-idx', type=int, default=1, metavar='N', help="start idx for numpy data file")
parser.add_argument('--end-idx', type=int, default=8, metavar='N', help="end idx for numpy data file")
parser.add_argument('--file-root', type=str, default="data", metavar='N', help="file name without extension")
parser.add_argument('--file-ext', type=str, default="npz", metavar='N', help="file extension")
parser.add_argument('--test-split', type=int, default=0.1, metavar='N', help="percent of data to use for test")
parser.add_argument('--random-seed', type=int, default=random.randint(1, 100), metavar='N', help="random seed")
args = parser.parse_args()

X_arrs = []
y_arrs = []

for i in range(args.start_idx, args.end_idx + 1):
    filepath = "data/{}{}.{}".format(args.file_root, i, args.file_ext)
    arr = np.load(filepath)
    print("Proccessing file '{}'".format(filepath))
    X = arr['x']
    y = arr['y']
    
    def x_conv(thing):
        idx = thing[0]

        reshape_x = lambda n: n.reshape((n.shape[2], n.shape[0], n.shape[1]))
        X_prev = reshape_x(X[idx - 1])
        X_temp = reshape_x(thing[1])

        if idx == 0:
            print("First Element")

        return [X_prev, X_temp] if idx else [X_temp, X_temp]
    
    X = np.array(list(map(x_conv, list(enumerate(X)))))
    X_arrs.append(X)
    y_arrs.append(y)

X_arrs = np.concatenate(X_arrs)
y_arrs = np.concatenate(y_arrs)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_arrs, y_arrs, test_size=args.test_split)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

out_filename = "data/{}_all.{}".format(args.file_root, args.file_ext)
np.savez(out_filename, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)