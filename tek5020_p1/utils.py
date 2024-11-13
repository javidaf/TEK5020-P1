import itertools
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from itertools import combinations
def load_dataset(file_path):
    data = pd.read_csv(file_path, sep='\s+', header=None)
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    return X, y

def error_rate_estimate(predictions, test_y):
    return np.mean(predictions != test_y)

def plot_dataset(X, y, title):
    classes = np.unique(y)
    markers = ['o', 's']
    colors = ['r', 'b']
    
    plt.figure(figsize=(8, 6))
    
    for cls, marker, color in zip(classes, markers, colors):
        plt.scatter(X[y == cls, 0], X[y == cls, 1],
                    marker=marker, color=color, label=f'Class {cls}')
    
    plt.title(f'Dataset: {title}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)


def plot_datasets_subfigures(datasets):
    num_datasets = len(datasets)
    cols = 2
    rows = (num_datasets + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
    axes = axes.flatten()

    for idx, (name, file_path) in enumerate(datasets.items()):
        X, y = load_dataset(file_path)
        ax = axes[idx]
        classes = np.unique(y)
        markers = ['o', 's']
        colors = ['r', 'b']

        for cls, marker, color in zip(classes, markers, colors):
            ax.scatter(X[y == cls, 0], X[y == cls, 1],
                       marker=marker, color=color, label=f'Class {cls}')

        ax.set_title(f'Dataset: {name}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        ax.grid(True)

        if X.shape[1] > 2:
            print(f"Dataset {name} has more than 2 features. Only the first two are plotted.")

    # Hide any unused subplots
    for idx in range(num_datasets, len(axes)):
        fig.delaxes(axes[idx])

    plt.show()


def plot_all_feature_combinations(file_path):
    """
    Plots all possible feature pair combinations for a given dataset.

    Parameters:
    - X: numpy.ndarray
        Feature matrix.
    - y: numpy.ndarray
        Class labels.
    - dataset_name: str
        Name of the dataset for the plot titles.
    """
    X,y = load_dataset(file_path)
    num_features = X.shape[1]
    feature_indices = range(num_features)
    combinations = list(itertools.combinations(feature_indices, 2))
    num_combinations = len(combinations)

    # Determine subplot grid size
    cols = 3
    rows = (num_combinations + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    markers = ['o', 's']
    colors = ['r', 'b']

    for idx, (i, j) in enumerate(combinations):
        ax = axes[idx]
        classes = np.unique(y)

        for cls, marker, color in zip(classes, markers, colors):
            ax.scatter(X[y == cls, i], X[y == cls, j],
                       marker=marker, color=color, label=f'Class {cls}')

        ax.set_title(f'{file_path} - Feature {i} vs Feature {j}')
        ax.set_xlabel(f'Feature {i}')
        ax.set_ylabel(f'Feature {j}')
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for idx in range(num_combinations, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()