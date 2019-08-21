import numpy as np

file = np.load("data/data_all.npz")

X_train, X_test, y_train, y_test = file['X_train'], file['X_test'], file['y_train'], file['y_test']
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)