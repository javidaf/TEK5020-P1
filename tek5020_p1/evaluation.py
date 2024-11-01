from classifiers import MinimumErrorRateClassifier, LeastSquaresClassifier, NearestNeighborClassifier
from preprocessing import train_test_split
import numpy as np

def evaluate_classifiers(X, y, feature_indices):
    # Select features
    X_selected = X[:, feature_indices]
    
    train_X, train_y, test_X, test_y = train_test_split(X_selected, y)
    
    MER = MinimumErrorRateClassifier()
    LS = LeastSquaresClassifier()
    NN = NearestNeighborClassifier()
    
    MER.fit(train_X, train_y)
    LS.fit(train_X, train_y)
    NN.fit(train_X, train_y)
    
    pred_MER = MER.predict(test_X)
    pred_LS = LS.predict(test_X)
    pred_NN = NN.predict(test_X)
    
    error_MER = np.mean(pred_MER != test_y)
    error_LS = np.mean(pred_LS != test_y)
    error_NN = np.mean(pred_NN != test_y)
    
    return {'MER': error_MER, 'LS': error_LS, 'NN': error_NN}