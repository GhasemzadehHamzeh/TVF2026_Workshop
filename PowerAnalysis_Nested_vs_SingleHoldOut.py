# Machine Learning Pitfalls in Voice Research – Workshop Code

# Author: Hamzeh Ghasemzadeh, PhD
# Affiliation: Communication Sciences and Disorders, University of Central Florida
# Lab: Voice and Speech Bioinformatics (VSI) Lab
# Email: hamzeh.ghasemzadeh@ucf.edu

# Description:
# This code was developed for a workshop presented at the Voice Foundation Symposium (TVF).
# It is part of a series of examples demonstrating common pitfalls in machine learning,
# their impact on results, and recommended practices to avoid them.

# Version: 1.0
# Year: 2026


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# -----------------------------
# 1. Dataset generation
# -----------------------------
def generate_synthetic_dataset(n_pairs=50, n_features=20, M=1.0, random_state=42):
    np.random.seed(random_state)

    n = n_pairs

    X_neg = np.random.normal(0, 1, (n, n_features))
    X_pos_main = np.random.normal(0, 1, (n, n_features - 2))
    X_pos_signal = np.random.normal(M, 1, (n, 2))
    X_pos = np.hstack((X_pos_main, X_pos_signal))

    y_neg = np.zeros(n)
    y_pos = np.ones(n)

    X = np.vstack((X_neg, X_pos))
    y = np.concatenate((y_neg, y_pos))

    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


# -----------------------------
# 2. Forward selection
# -----------------------------
def forward_selection_fixed_k(X, y, k_features, eval_func):
    n_features = X.shape[1]

    selected = []
    remaining = list(range(n_features))

    while len(selected) < k_features:
        best_feature = None
        best_score = -np.inf

        for f in remaining:
            candidate = selected + [f]
            score = eval_func(candidate)

            if score > best_score:
                best_score = score
                best_feature = f

        selected.append(best_feature)
        remaining.remove(best_feature)

    return selected


# -----------------------------
# 3. Hold-out forward selection
# -----------------------------
def forward_selection_holdout(X_train, y_train, X_test, y_test, k_features):

    def eval_func(features):
        model = LogisticRegression(max_iter=100)
        model.fit(X_train[:, features], y_train)

        return accuracy_score(
            y_test,
            model.predict(X_test[:, features])
        )

    return forward_selection_fixed_k(X_train, y_train, k_features, eval_func)


# -----------------------------
# 4. Nested forward selection
# -----------------------------
def forward_selection_nested(X, y, inner_cv, k_features):

    def eval_func(features):
        fold_scores = []

        for train_idx, val_idx in inner_cv.split(X, y):
            X_train = X[train_idx]
            y_train = y[train_idx]

            X_val = X[val_idx]
            y_val = y[val_idx]

            model = LogisticRegression(max_iter=100)
            model.fit(X_train[:, features], y_train)

            acc = accuracy_score(
                y_val,
                model.predict(X_val[:, features])
            )

            fold_scores.append(acc)

        return np.mean(fold_scores)  

    return forward_selection_fixed_k(X, y, k_features, eval_func)


# -----------------------------
# 5. Hold-out evaluation
# -----------------------------
def holdout_evaluation(X, y, k_features):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    selected = forward_selection_holdout(
        X_train, y_train, X_test, y_test, k_features
    )

    model = LogisticRegression(max_iter=100)
    model.fit(X_train[:, selected], y_train)

    return accuracy_score(y_test, model.predict(X_test[:, selected]))


# -----------------------------
# 6. Nested CV evaluation
# -----------------------------
def nested_cv_evaluation(X, y, k_features, n_splits=5):
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    outer_scores = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        selected = forward_selection_nested(
            X_train_outer, y_train_outer, inner_cv, k_features
        )

        model = LogisticRegression(max_iter=100)
        model.fit(X_train_outer[:, selected], y_train_outer)

        outer_acc = accuracy_score(
            y_test_outer,
            model.predict(X_test_outer[:, selected])
        )

        outer_scores.append(outer_acc)

    return np.mean(outer_scores)



# -----------------------------
# 7. Main experiment
# -----------------------------
if __name__ == "__main__":

    n_reps = 25
    k_features = 2
    
    resultsHoldOut_M0 = []
    resultsHoldOut_M1 = []
    
    resultsNested_M0 = []
    resultsNested_M1 = []

    # ---------- M = 0 (Null) ----------
    for seed in range(n_reps):
        X, y = generate_synthetic_dataset(
            n_pairs=50,
            n_features=20,
            M=0,
            random_state=seed
        )

        acc = holdout_evaluation(X, y, k_features)
        resultsHoldOut_M0.append(acc)        

        acc = nested_cv_evaluation(X, y, k_features)
        resultsNested_M0.append(acc)


    # ---------- M = 0.5 (Alternative) ----------
    for seed in range(n_reps):
        X, y = generate_synthetic_dataset(
            n_pairs=50,
            n_features=20,
            M=0.5,
            random_state=seed
        )

        acc = holdout_evaluation(X, y, k_features)
        resultsHoldOut_M1.append(acc)        

        acc = nested_cv_evaluation(X, y, k_features)
        resultsNested_M1.append(acc)

    # -----------------------------
    # Plot histograms (Hold-out vs Nested CV)
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # ---- Hold-out subplot ----
    axes[0].hist(resultsHoldOut_M0, bins=25,range=(0, 1), alpha=0.6, label='M = 0 (Null)')
    axes[0].hist(resultsHoldOut_M1, bins=25,range=(0, 1), alpha=0.6, label='M = 0.5 (Alternative)')
    axes[0].set_xlabel('Accuracy')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Hold-out')
    axes[0].legend()

    # ---- Nested CV subplot ----
    axes[1].hist(resultsNested_M0, bins=25,range=(0, 1), alpha=0.6, label='M = 0 (Null)')
    axes[1].hist(resultsNested_M1, bins=25,range=(0, 1), alpha=0.6, label='M = 0.5 (Alternative)')
    axes[1].set_xlabel('Accuracy')
    axes[1].set_title('Nested CV')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

