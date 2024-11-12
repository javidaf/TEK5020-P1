import numpy as np


def estimate_prior_probabilities(y):
    classes, counts = np.unique(y, return_counts=True)
    prior_probabilities = counts / counts.sum()
    priors = dict(zip(classes, prior_probabilities))
    formatted_priors = {int(cls): float(prob) for cls, prob in priors.items()}
    return formatted_priors


class MinimumErrorRateClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = {}
        total_samples = len(y)

        for cls in self.classes:
            X_cls = X[y == cls]
            mean = np.mean(X_cls, axis=0)
            covariance = np.cov(X_cls, rowvar=False)
            covariance = np.atleast_2d(covariance)
            cov_inv = np.linalg.inv(covariance)
            prior = len(X_cls) / total_samples

            W = -0.5 * cov_inv
            w = cov_inv @ mean
            w0 = -0.5 * mean.T @ cov_inv @ mean - 0.5 * np.log(np.linalg.det(covariance)) + np.log(prior)

            self.parameters[cls] = {'W': W, 'w': w, 'w0': w0}

    def predict(self, X):
        scores = np.zeros((X.shape[0], len(self.classes)))
        for idx, cls in enumerate(self.classes):
            W = self.parameters[cls]['W']
            w = self.parameters[cls]['w']
            w0 = self.parameters[cls]['w0']
            scores[:, idx] = np.sum(X @ W * X, axis=1) + X @ w + w0
        predictions = self.classes[np.argmax(scores, axis=1)]
        return predictions


class LeastSquaresClassifier:
    def __init__(self):
        self.a = None

    def fit(self, X, y):
        Y = np.hstack((np.ones((X.shape[0], 1)), X))       
        b = np.where(y == 1, 1, -1)
        
        YTY = Y.T @ Y
        YTb = Y.T @ b
        self.a = np.linalg.inv(YTY) @ YTb

    def predict(self, X):
        Y = np.hstack((np.ones((X.shape[0], 1)), X))
        scores = Y @ self.a
        predictions = np.where(scores > 0, 1, 2)
        return predictions

class NearestNeighborClassifier:
    def __init__(self):
        self.train_X = None
        self.train_y = None

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X):
        predictions = []
        for x in X:
            # Euclidean distances to all training samples
            distances = np.linalg.norm(self.train_X - x, axis=1)
            # index of the nearest neighbor
            nearest_idx = np.argmin(distances)
            # Assign the class label of the nearest neighbor
            predictions.append(self.train_y[nearest_idx])
        return np.array(predictions)

