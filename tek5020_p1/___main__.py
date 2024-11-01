import itertools

from matplotlib import pyplot as plt
from tek5020_p1.utils import load_dataset
from tek5020_p1.preprocessing import train_test_split
from tek5020_p1.classifiers import (MinimumErrorRateClassifier,
                                    LeastSquaresClassifier,
                                    NearestNeighborClassifier,
                                    estimate_prior_probabilities)
import numpy as np
from evaluation import evaluate_classifiers
from tek5020_p1.utils import plot_dataset, plot_datasets_subfigures,plot_all_feature_combinations

file = 'TEK5020-P1\data\ds-1.txt'




X, y = load_dataset(file)

train_X , train_y, test_X, test_y = train_test_split(X, y) 

MER_classifier = MinimumErrorRateClassifier()
LS_classifier = LeastSquaresClassifier()
NN_classifier = NearestNeighborClassifier()

MER_classifier.fit(train_X, train_y)
predictions = MER_classifier.predict(test_X)
error_rate = np.mean(predictions != test_y)

LS_classifier.fit(train_X, train_y)
predictions = LS_classifier.predict(test_X)
error_rate_ls = np.mean(predictions != test_y)

NN_classifier.fit(train_X, train_y)
predictions = NN_classifier.predict(test_X)
error_rate_nn = np.mean(predictions != test_y)

priors = estimate_prior_probabilities(train_y)
print("Prior Probabilities:", priors)
print("Minimum Error Rate Classifier Error Rate:", error_rate)
print("Least Squares Classifier Error Rate:", error_rate_ls)
print("Nearest Neighbor Classifier Error Rate:", error_rate_nn)

datasets = {
    'Dataset_1': 'TEK5020-P1/data/ds-1.txt',
    'Dataset_2': 'TEK5020-P1/data/ds-2.txt',
    'Dataset_3': 'TEK5020-P1/data/ds-3.txt'
}

for name, file_path in datasets.items():
    print(f"{name}")
    X, y = load_dataset(file_path)
    num_features = X.shape[1]
    
    # Feature combination rankings based on Nearest Neighbor
    feature_rankings = {}
    for d in range(1, num_features + 1):
        combinations = itertools.combinations(range(num_features), d)
        error_list = []
        for combo in combinations:
            errors = evaluate_classifiers(X, y, combo)
            error_nn = errors['NN']
            error_list.append({'features': combo, 'error_NN': error_nn})
        
        # Sort feature combinations by NN error rate
        sorted_errors = sorted(error_list, key=lambda x: x['error_NN'])
        
        # Get best, second best, and worst feature combinations
        best_combo = sorted_errors[0]
        second_best_combo = sorted_errors[1] if len(sorted_errors) > 1 else None
        worst_combo = sorted_errors[-1]
        
        feature_rankings[d] = {
            'best': best_combo,
            'second_best': second_best_combo,
            'worst': worst_combo
        }
        
        print(f"Feature dimension d={d}:")
        print(f"  Best combination: {best_combo['features']} with NN error rate {best_combo['error_NN']}")
        if second_best_combo:
            print(f"  Second best combination: {second_best_combo['features']} with NN error rate {second_best_combo['error_NN']}")
        print(f"  Worst combination: {worst_combo['features']} with NN error rate {worst_combo['error_NN']}")
    
    # Classifier rankings for best feature combinations
    for d, info in feature_rankings.items():
        best_features = info['best']['features']
        errors = evaluate_classifiers(X, y, best_features)
        
        # Sort classifiers by error rate
        sorted_classifiers = sorted(errors.items(), key=lambda x: x[1])
        
        # Get best, second best, and worst classifiers
        best_classifier = sorted_classifiers[0]
        second_best_classifier = sorted_classifiers[1] if len(sorted_classifiers) > 1 else None
        worst_classifier = sorted_classifiers[-1]
        
        print(f"\nClassifier rankings for feature dimension d={d}:")
        print(f"  Best classifier: {best_classifier[0]} with error rate {best_classifier[1]}")
        if second_best_classifier:
            print(f"  Second best classifier: {second_best_classifier[0]} with error rate {second_best_classifier[1]}")
        print(f"  Worst classifier: {worst_classifier[0]} with error rate {worst_classifier[1]}")
    print("\n")

plot_all_feature_combinations(datasets['Dataset_3'])
