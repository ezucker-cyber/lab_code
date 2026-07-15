"""
Generic model evaluation: holdout split, fit, and AIC/BIC/accuracy scoring
for any single model. Model-agnostic — hand it a fit function and a
predict_proba function (both from fit_integrators.py, or your own), and
it does the rest. Call once per model, then collect results into a
DataFrame with build_model_results_df().
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Holdout split — call once, reuse the same split across all models so the
# comparison is apples-to-apples.
# ---------------------------------------------------------------------------

def split_holdout(d_matrix, y, holdout_frac=0.2, seed=None):
    """
    Randomly split trials into train/test sets.

    holdout_frac : fraction of trials held out for testing (e.g. 0.2 = 20%)
    seed         : optional int for a reproducible split

    Returns: d_train, y_train, d_test, y_test
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(round(n * holdout_frac))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return d_matrix[train_idx], y[train_idx], d_matrix[test_idx], y[test_idx]


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(fit_fn, predict_proba_fn, d_train, y_train, d_test, y_test):
    """
    Fit one model on (d_train, y_train), score it on (d_test, y_test).

    fit_fn           : one of fit_perfect / fit_leaky / fit_confirmation /
                       fit_normative (or any function with the same
                       signature: fit_fn(d_matrix, y) -> dict with
                       'params' and 'k').
    predict_proba_fn : matching predict_proba_* function:
                       predict_proba_fn(params, d_matrix) -> P(choose left)
    d_train, y_train : training data (pass the full dataset for both
                       train and test if you don't want a holdout split)
    d_test, y_test   : evaluation data

    Returns dict: params, k, ll, aic, bic, accuracy, n
    """
    fit_result = fit_fn(d_train, y_train)
    params, k = fit_result['params'], fit_result['k']

    p = np.clip(predict_proba_fn(params, d_test), 1e-10, 1 - 1e-10)
    ll = np.sum(y_test * np.log(p) + (1 - y_test) * np.log(1 - p))
    n = len(y_test)

    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(n)
    accuracy = np.mean((p > 0.5).astype(int) == y_test)

    return {'params': params, 'k': k, 'll': ll, 'aic': aic, 'bic': bic,
            'accuracy': accuracy, 'n': n}


# ---------------------------------------------------------------------------
# Collect however many evaluate_model() calls you made into one DataFrame
# ---------------------------------------------------------------------------

def build_model_results_df(results_by_model):
    """
    results_by_model : dict {model_name: result_dict}, where each
                       result_dict is whatever evaluate_model() returned.

    Returns: DataFrame indexed by model name, columns
             [params, k, ll, aic, bic, accuracy, n]
    """
    return pd.DataFrame.from_dict(results_by_model, orient='index')[
        ['params', 'k', 'll', 'aic', 'bic', 'accuracy', 'n']
    ]