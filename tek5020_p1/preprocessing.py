
def train_test_split(X,y):
    train_X = X[::2]  # Objects 1, 3, 5, ...
    train_y = y[::2]
    test_X = X[1::2]  # Objects 2, 4, 6, ...
    test_y = y[1::2]

    return train_X, train_y, test_X, test_y


    