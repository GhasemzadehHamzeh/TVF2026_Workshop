import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# -----------------------------
# 1. Dataset generation
# -----------------------------
def generate_synthetic_dataset(n_pos=50, imbalance_ratio=1.0, n_features=20, M=1.0, random_state=42):
    np.random.seed(random_state)

    n_neg = int(n_pos * imbalance_ratio)

    X_neg = np.random.normal(0, 1, (n_neg, n_features))
    X_pos_main = np.random.normal(0, 1, (n_pos, n_features - 2))
    X_pos_signal = np.random.normal(M, 1, (n_pos, 2))
    X_pos = np.hstack((X_pos_main, X_pos_signal))

    y_neg = np.zeros(n_neg)
    y_pos = np.ones(n_pos)

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

    n_reps = 5  
    n_pos = 30
    n_features = 10
    M = 1
    k_features = 2
    
    imbalance_percents = np.arange(100, 500, 50)   # 100% to 200%
    imbalance_ratios = imbalance_percents / 100.0

    holdout_means = []
    holdout_stds = []

    nested_means = []
    nested_stds = []

    for ratio in imbalance_ratios:

        holdout_results = []
        nested_results = []

        for seed in range(n_reps):

            X, y = generate_synthetic_dataset(
                n_pos=n_pos,
                imbalance_ratio=ratio,
                n_features=n_features,
                M=M,
                random_state=seed
            )

            holdout_results.append(holdout_evaluation(X, y, k_features))
            nested_results.append(nested_cv_evaluation(X, y, k_features))

        holdout_means.append(np.mean(holdout_results))
        holdout_stds.append(np.std(holdout_results))

        nested_means.append(np.mean(nested_results))
        nested_stds.append(np.std(nested_results))

    # -----------------------------
    # Plot error bars
    # -----------------------------
    plt.figure(figsize=(8, 5))

    plt.errorbar(
        imbalance_percents,
        holdout_means,
        yerr=holdout_stds,
        label='Hold-out',
        capsize=3
    )

    plt.errorbar(
        imbalance_percents,
        nested_means,
        yerr=nested_stds,
        label='Nested CV',
        capsize=3
    )

    plt.xlabel('Imbalance Ratio (% of negatives vs positives)')
    plt.ylabel('Accuracy (mean ± std)')
    plt.title('Effect of Class Imbalance (M = 1)')
    plt.legend()

    plt.tight_layout()
    plt.show()