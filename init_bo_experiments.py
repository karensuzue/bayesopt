"""
This file runs one replicate.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import openml

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score


def vector_to_param_dict(values, hyperparam_names, config):
    """
    Converts a list of hyperparameter values into a dictionary using the provided configuration.
    
    Parameters:
    - values: List of values, one for each hyperparameter.
    - hyperparam_names: List of hyperparameter names.
    - config: Dictionary mapping each hyperparameter to its bounds and type.

    Returns:
    - A dictionary mapping hyperparameter names to properly typed values.
    """
    assert len(hyperparam_names) == len(values)
    param_dict = {}
    for name, value in zip(hyperparam_names, values):
        param_type = config[name]['type']
        if param_type == 'int':
            param_dict[name] = int(round(value))
        elif param_type == 'float':
            param_dict[name] = float(value)
        else:
            raise ValueError("Unsupported type: ", param_type)
    return param_dict


# We're trying to MAXIMIZE this objective function
def objective(X, hyperparam_names, hyperparam_config, X_train, y_train, seed):
    """
    Evaluates a set of hyperparameter configurations using cross-validated accuracy.

    Parameters:
    - X: array-like of shape (n_samples, n_features)
      Each row represents a set of hyperparameter values, e.g., x_n = [n_estimators, max_depth, ...]

    Returns:
    - An array of shape (n_samples, 1) containing the cross-validated accuracy scores 
      for each hyperparameter configuration.
    """
    scores = []
    for param in X: # iterate through each input
        # convert each row of X into a dictionary with appropriate types 
        param_dict = vector_to_param_dict(param, hyperparam_names, hyperparam_config)
        model = RandomForestClassifier(**param_dict, random_state=seed)
        # No need to invert, already maximizing - higher accuracy value is better 
        score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean() 
        scores.append(score)
    return np.array(scores).reshape(-1,1)
    

def squared_euclidean_distance(xa, xb):
    """
    Computes the pairwise squared Euclidean distances between two sets of vectors.
    The squared Euclidean distance between two vectors a and b is defined as:
      ||a - b||^2 = ||a||^2 + ||b||^2 - 2ab

    Parameters:
    - xa : array of shape (n_samples_a, n_features)
      First set of input vectors.
    - xb : array of shape (n_samples_b, n_features)
      Second set of input vectors.
    
    Returns:
    - An array of shape (n_samples_a, n_samples_b), representing the matrix of 
      squared Euclidean distances between each pair of vectors from 'xa' and 'xb'.
    """
    # Reduce (n,d) to (n,1), (m,d) to (1,m)
    xa2 = np.sum(xa**2, axis=1).reshape(-1, 1)   # shape: (n,1)
    xb2 = np.sum(xb**2, axis=1).reshape(1, -1)   # shape: (1,m)

    # print("xa2:", xa2)
    # print("xb2:", xb2)

    # Compute pairwise dot products
    dot_prod = np.dot(xa, xb.T)                  # shape: (n,m)

    # Apply the distance formula
    return xa2 + xb2 - 2 * dot_prod              # shape: (n,m)


def rbf_kernel(xa, xb, sigma=1):
    """
    Computes the Radial Basis Function (RBF) Kernel between two sets of input vectors.
    
    Parameters:
    - xa : array of shape (n_samples_a, n_features)
      First set of input vectors.
    - xb : array of shape (n_samples_b, n_features)
      Second set of input vectors.
    - sigma: float, the length scale parameter of the RBF kernel. Smaller values lead to narrower kernels.
      
    Returns:
    - An array of shape (n_samples_a, n_samples_b), representing the kernel matrix, where 
      each entry (i, j) represents the similarity between xa[i] and xb[j] based on the squared Euclidean distance.
    """
    xa = np.array(xa)
    xb = np.array(xb)
    sqdist = squared_euclidean_distance(xa, xb)
    return np.exp(-0.5 * 1/(sigma**2) * sqdist)


def GP(X1, y1, X2, sigma=1.0, noise=0):
    """
    Compute the posterior mean and covariance of a Gaussian Process (GP) using the RBF Kernel. 

    Parameters:
    - X1: array of shape (n_samples_train, n_features), representing the training input data.
    - y1: array of shape (n_samples_train, 1), representing the training target values. 
    - X2: array of shape (n_samples_test, n_features), representing the test input data where
      predictions are to be made.
    - sigma: float, length scale parameter for the RBF kernel. Controls the smoothness of the function.
    - noise: float, standard deviation of Gaussian noise added to the training targets. 
      Helps model noisy observations.
      
    Returns:
    - mu: array of shape (n_samples_test, 1), representing the posterior mean predictions for the test inputs.
    - cov: array of shape (n_samples_test, n_samples_test), the posterior covariance matrix 
      for representing uncertainty in the predictions.

    Notes:
    - The function assumes a zero-mean GP prior.
    """
    # In case 1D array passed
    y1 = y1.reshape(-1, 1)
    
    # Kernel (covariance) matrices
    # Add noise on diagonal for K11
    K11 = rbf_kernel(X1, X1, sigma) + (noise**2) * np.eye(len(X1)) + 1e-8 * np.eye(len(X1)) # shape: (n,n)
    K22 = rbf_kernel(X2, X2, sigma)                                # shape: (m,m)
    K12 = rbf_kernel(X1, X2, sigma)                                # shape: (n,m)

    # Invert K11
    K11_inv = np.linalg.inv(K11)

    # Posterior mean
    mu = K12.T @ K11_inv @ y1 # # @ is matrix multiplication

    # Posterior covariance
    cov = K22 - K12.T @ K11_inv @ K12

    # Regularize to ensure diagonal values are non-negative
    cov += 1e-6 * np.eye(cov.shape[0])

    return mu, cov


# Acquisition function: UCB
def acquisition_ucb(mu, std, beta=2.0):
    """
    Computes the Upper Confidence Bound (UCB) acquisition function for Bayesian Optimization (BO).

    Parameters:
    - mu: array of shape (n_samples, 1), representing the predicted mean values 
      from the GP for each candidate input.
    - std: array of shape (n_samples, 1), representing the predicted standard deviations (uncertainty)
      from the GP for each candidate input.
    - beta: float, exploration-exploitation trade-off parameter. Higher values encourage exploration by
      giving more weight to uncertainty. 
    """
    return mu + beta * std

def bayesian_opt(objective, hyperparam_names, hyperparam_config, X_train, y_train, seed,
                 init_points=5, iterations=10, beta=2.0, sigma=1.0, noise=0):
    """
    Performs Bayesian Optimization (BO) using a GP surrogate model and 
    the UCB acquisition function.

    Parameters:
    - objective: callable, the objective function to be maximized. It should accept a 2D array of 
      hyperparameter configurations and return a 2D array of scores
    - init_points: int, number of initial random samples to evaluate as training data, before starting the 
      optimization loop.
    - iteration: int, number of optimization steps to perform after initial sampling.
    - beta: float, exploration-exploitation trade-off parameter for the UCB acquisition function.
      Higher values encourage exploration.
    - sigma: float, length scale parameter for the RBF kernel used in the GP.
    - noise: float, standard deviation of Gaussian noise added to observations (from the latent objective). 

    Returns:
    - best_hyperparam: array of shape (1, n_features), representing the best hyperparameter configuration found
      during the optimization process. 

    Notes:
    - For plotting and scoring (acquisition), the function uses a grid consisting of 100 points 
      along each dimension.
    """
    
    # === Step 1: Generate initial training data ===
    # Randomly sample 'init_points' configurations from the hyperparameter space
    X_param_train = []
    for i in range(init_points):
        sample = [np.random.uniform(*param['bounds']) for param in hyperparam_config.values()]
        X_param_train.append(sample)
    # print("X train: ", X_param_train)
    
    # Evaluate the objective function on the initial samples
    y_param_train = objective(X_param_train, hyperparam_names, hyperparam_config, X_train, y_train, seed)
    # print("y train: ", y_param_train)


    # === Step 2: Create a grid for acquisition function evaluation ===
    # Generate 100 evenly spaced values for each hyperparameter dimension
    dimensions = []
    for param in hyperparam_config.values():
        dimensions.append(np.linspace(*param['bounds'], 100))
    
    # Create a meshgrid of all combinations of hyperparameter values
    grids = np.meshgrid(*dimensions) 
    
    # Flatten the grid into a 2D array of shape (100*num_dimensions, num_dimensions)
    X_grid = np.vstack([grid.ravel() for grid in grids]).T
    # print("X_grid: ", X_grid)

    # === Step 3: Bayesian Optimization Loop ===
    mu = None # Posterior mean predictions from GP
    cov = None # Posterior covariance matrix from GP
    std = None # Standard deviation (uncertainty) for each prediction from GP
    best_hyperparam = None # Best hyperparameter configuration found
    best_cv_score = float('-inf') # Best CV score seen so far

    for i in range(iterations):
        # print("Iteration ", i)
        # Fit GP surrogate model and predict on X_grid
        mu, cov = GP(X_param_train, y_param_train, X_grid, sigma, noise)
        # print("Mean: ", mu)
        # print("Covariance: ", cov)

        # Obtain standard deviation (uncertainty)
        std = np.sqrt(np.diag(cov)).reshape(-1, 1)

        # Evaluate acquisition function on X_grid
        scores = acquisition_ucb(mu, std)
        # print("Scores: ", scores)
        # print("Scores shape: ", scores.shape)

        # Select next point to observe
        next_x = np.array([X_grid[np.argmax(scores)]]) # maximize acquisition function
        # print("Next x: ", next_x)
        next_y = objective(next_x, hyperparam_names, hyperparam_config, X_train, y_train, seed)
        # print("Next y: ", next_y)

        # Update best score and hyperparameter configuration
        if next_y[0,0] > best_cv_score:
            best_cv_score = next_y[0,0]
            best_hyperparam = next_x

        # Add new data point to training set
        X_param_train = np.vstack((X_param_train, next_x.reshape(1, -1)))
        y_param_train = np.vstack((y_param_train, next_y.reshape(1, -1)))

    return best_hyperparam

def main():
    parser = argparse.ArgumentParser()
    # bo_sigma (RBF kernel parameter)
    parser.add_argument('--dataset', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--method', choices=['bo_5', 'bo_10', 'bo_20', 'bo_50', 'random', 'default'], required=True)
    parser.add_argument('--replicate', type=int, required=True) # this is basically a seed
    parser.add_argument('--evaluations', type=int, required=True) # number of evaluations on the objective function
    args = parser.parse_args()

    dataset_idx = args.dataset
    method = args.method
    replicate = args.replicate
    evaluations = args.evaluations
    np.random.seed(replicate) # makes sure random numbers are reproducible

    hyperparam_names = ['n_estimators', 'max_depth'] # 'min_samples_split'
    hyperparam_config = {'n_estimators': {'bounds': (50, 500), 'type': 'int'},
                     'max_depth': {'bounds': (10, 100), 'type': 'int'}}
                     # 'min_samples_split': {'bounds': (0.01, 0.5), 'type': 'float'}}

    # 3 datasets: blood-transfusion-service-center (1464), phoneme (1489), spambase (44)
    openml_tasks = [1464, 1489, 44]
    if len(openml_tasks) <= dataset_idx:
        raise ValueError(f"dataset={dataset_idx} does not exist")
    dataset = openml.datasets.get_dataset(openml_tasks[dataset_idx])
    df, *_ = dataset.get_data()

    # Fix inconsistent naming in dataset 44
    if dataset_idx == 2:
        df.rename(columns={'class': 'Class'}, inplace=True)

    X = df.drop(columns='Class').values 
    y = df['Class'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=replicate)

    # Bayesian Optimization
    if method.startswith("bo"):
        init_points = 50 # points to sample initially
        if evaluations <= init_points:
            raise ValueError(f"evaluations={evaluations} must be greater than init_points={init_points}")
        iters = evaluations - init_points # 1 evaluation per iteration
        sigma = float(re.findall(r'\d+', method)[0]) # RBF kernel parameter
        best_params = bayesian_opt(objective, hyperparam_names, hyperparam_config, X_train, y_train, replicate,
                                   init_points=init_points, iterations=iters, beta=2.0, sigma=sigma)
        param_dict = vector_to_param_dict(best_params[0], hyperparam_names, hyperparam_config)

    # Random search through the hyperparameter space
    elif method == "random": 
        best_params = None
        best_cv_score = float('-inf')
        for i in range(evaluations):
            # Generate a random hyperparameter configuration
            params = np.array([[np.random.uniform(*param['bounds']) for param in hyperparam_config.values()]])
            # Compute cross-validated accuracy
            cv_score = objective(params, hyperparam_names, hyperparam_config, X_train, y_train, replicate)
            # Update best_parameter
            if cv_score[0,0] > best_cv_score:
                best_params = params
                best_cv_score = cv_score
        param_dict = vector_to_param_dict(best_params[0], hyperparam_names, hyperparam_config)
    
    # Use default hyperparameters (baseline, no search)
    elif method == "default":
        param_dict = {}

    # Retrain model using best param
    model = RandomForestClassifier(**param_dict, random_state=replicate)
    model.fit(X_train, y_train)

    final_score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

    # Predict on held-out test set
    y_pred = model.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)

    # Save results
    result = {
        "dataset": dataset_idx,
        "replicate": replicate,
        "evaluations": evaluations,
        "method": method,
        "final_cv_accuracy_score": final_score,
        "test_accuracy_score": test_score,
        "best_params": model.get_params() if method == "default" else param_dict
    }

    out_dir = Path(f"bo_experiments/{dataset_idx}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/f"{method}_{replicate}.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()