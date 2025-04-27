import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical  

def custom_train_test_split(X, Y, test_size):
    n_samples = len(X)
    n_test_samples = int(n_samples * test_size)
    indices = np.arange(n_samples)
    test_indices = np.random.choice(indices, size=n_test_samples, replace=False)
    train_indices = np.setdiff1d(indices, test_indices)
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]
    return X_train, X_test, Y_train, Y_test

data = pd.read_csv(r"\Data\breastcancer.csv")
print(data.head())

X = data.iloc[:, 0:-1]
Y = data.iloc[:, -1]

Y = [label - 1 for label in Y]
Y = to_categorical(Y, num_classes=2)

X_train, X_test, Y_train, Y_test = custom_train_test_split(X, Y, test_size=0.30)

print("All Data:", data.shape)
print("Train Data:", X_train.shape)
print("Test Data:", X_test.shape)
print("Train out Data:", Y_train.shape)
print("Test out Data:", Y_test.shape)
##
