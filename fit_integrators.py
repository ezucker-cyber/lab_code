"""
Per-subject maximum likelihood fitting for the four integrator models.

Each model reduces the 12 frame differences d(t) = left(t) - right(t) for a
trial to a single decision variable (DV). Choice is then modeled as:

    P(choose left) = logistic(a + b * DV)

Perfect and confirmation-weighted are linear in their fitted parameters
(ordinary logistic regression — cheap, one-shot, no nesting). Leaky and
normative have a parameter (lambda, h) sitting inside DV itself, so they
require joint nonlinear optimization over all parameters at once.

Input convention for every fit_* function:
    d_matrix : (n_trials, n_frames) array, d_matrix[i, t] = left(t) - right(t)
    y        : (n_trials,) array, 1 = chose left, 0 = chose right
"""

import ast
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from integrators import psi


# ---------------------------------------------------------------------------
# Data loading: results.csv (from experiment.html) -> d_matrix, y
# ---------------------------------------------------------------------------

def load_trial_data(filepath, n_frames=12):
    """
    Parse a results.csv produced by experiment.html into the (d_matrix, y)
    format every fit_* function expects.

    Each left_t/right_t cell is stored as a string tuple "(value, switched)".
    We only need `value` here; `switched` isn't used for fitting.

    Returns:
        d_matrix : (n_trials, n_frames) array, d_matrix[i, t] = left(t) - right(t)
        y        : (n_trials,) array, 1 = chose left, 0 = chose right (from 'choice' col)
        df       : the parsed DataFrame, in case you want H, correct, etc.
    """
    df = pd.read_csv(filepath)

    def parse_value(cell):
        # cell looks like '(31.4002, 0)' or '"(31.4002, 0)"' after CSV round-trip
        cell = str(cell).strip().strip('"')
        value, _switched = ast.literal_eval(cell)
        return value

    d_matrix = np.zeros((len(df), n_frames))
    for t in range(1, n_frames + 1):
        left_vals  = df[f'left_{t}'].apply(parse_value).values
        right_vals = df[f'right_{t}'].apply(parse_value).values
        d_matrix[:, t - 1] = left_vals - right_vals

    y = df['choice'].values

    return d_matrix, y, df


# ---------------------------------------------------------------------------
# Shared likelihood machinery
# ---------------------------------------------------------------------------

def _neg_log_likelihood(params, X, y):
    """Generic negative log-likelihood for logistic regression on design matrix X."""
    linear = X @ params
    p = np.clip(expit(linear), 1e-10, 1 - 1e-10)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def _fit_linear_logistic(X, y, init):
    """Ordinary (linear-in-parameters) logistic MLE fit — no bounds needed."""
    return minimize(_neg_log_likelihood, init, args=(X, y), method='L-BFGS-B')


# ---------------------------------------------------------------------------
# 1. Perfect integrator
#    DV = sum_t d(t)
# ---------------------------------------------------------------------------

def fit_perfect(d_matrix, y):
    """
    Returns dict: params=[a, b], ll, n, k
    """
    DV = d_matrix.sum(axis=1)
    X = np.column_stack([np.ones(len(DV)), DV])

    result = _fit_linear_logistic(X, y, init=[0, 0.01])

    return {'params': result.x, 'll': -result.fun, 'n': len(y), 'k': 2}


def predict_proba_perfect(params, d_matrix):
    """P(choose left) for arbitrary d_matrix, given already-fitted params=[a, b]."""
    a, b = params
    DV = d_matrix.sum(axis=1)
    return expit(a + b * DV)


# ---------------------------------------------------------------------------
# 2. Leaky integrator
#    DV = sum_t lambda^(n_frames-1-t) * d(t)
#    lambda sits inside the exponent -> nonlinear joint optimization
# ---------------------------------------------------------------------------

def _neg_ll_leaky(params, d_matrix, y):
    a, b, lam = params
    n_frames = d_matrix.shape[1]
    weights = lam ** (n_frames - 1 - np.arange(n_frames))
    DV = d_matrix @ weights
    linear = a + b * DV
    p = np.clip(expit(linear), 1e-10, 1 - 1e-10)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_leaky(d_matrix, y, init=(0, 0.01, 0.9)):
    """
    Returns dict: params=[a, b, lambda], ll, n, k
    """
    bounds = [(None, None), (None, None), (1e-6, 1.0)]
    result = minimize(_neg_ll_leaky, init, args=(d_matrix, y),
                       method='L-BFGS-B', bounds=bounds)

    return {'params': result.x, 'll': -result.fun, 'n': len(y), 'k': 3}


def predict_proba_leaky(params, d_matrix):
    """P(choose left) for arbitrary d_matrix, given already-fitted params=[a, b, lambda]."""
    a, b, lam = params
    n_frames = d_matrix.shape[1]
    weights = lam ** (n_frames - 1 - np.arange(n_frames))
    DV = d_matrix @ weights
    return expit(a + b * DV)


# ---------------------------------------------------------------------------
# 3. Confirmation-weighted integrator (Glickman, Moran & Usher, 2022)
#    DV = X1 + phi * X2
#      X1 = sum_t d(t)
#      X2 = sum_t sign(d(t)) * i(t)     [i(t) = causal same-sign streak length]
#    phi multiplies a *fixed* regressor -> linear-in-parameters, no nesting.
#    Fit a, b1, b2 directly; recover phi = b2 / b1 afterward.
# ---------------------------------------------------------------------------

def _streak_regressor(d_row):
    """
    X2 for one trial: sum_t sign(d(t)) * i(t). i(t) = number of consecutive
    prior frames sharing the current frame's sign (causal, resets on flip).
    """
    streak = 0
    prev_sign = None
    total = 0.0
    for dt in d_row:
        cur_sign = np.sign(dt)
        if prev_sign is not None and cur_sign != 0 and cur_sign == prev_sign:
            streak += 1
        else:
            streak = 0
        total += cur_sign * streak
        prev_sign = cur_sign
    return total


def fit_confirmation(d_matrix, y):
    """
    Returns dict: params=[a, b1, b2], phi (= b2/b1), ll, n, k
    """
    X1 = d_matrix.sum(axis=1)
    X2 = np.array([_streak_regressor(row) for row in d_matrix])
    X = np.column_stack([np.ones(len(y)), X1, X2])

    result = _fit_linear_logistic(X, y, init=[0, 0.01, 0.0])
    a, b1, b2 = result.x
    phi = b2 / b1 if b1 != 0 else np.nan

    return {'params': result.x, 'phi': phi, 'll': -result.fun, 'n': len(y), 'k': 3}


def predict_proba_confirmation(params, d_matrix):
    """P(choose left) for arbitrary d_matrix, given already-fitted params=[a, b1, b2]."""
    a, b1, b2 = params
    X1 = d_matrix.sum(axis=1)
    X2 = np.array([_streak_regressor(row) for row in d_matrix])
    return expit(a + b1 * X1 + b2 * X2)


# ---------------------------------------------------------------------------
# 4. Normative integrator (Glaze, Kable & Gold, 2015)
#    L(t) = psi(L(t-1), h) + d(t),   DV = L(n_frames)
#    h sits inside the recursive psi() update -> nonlinear joint optimization
# ---------------------------------------------------------------------------

def _normative_dv(d_row, h):
    """Recursively compute the final belief L for one trial given hazard h."""
    L = 0.0
    for t, dt in enumerate(d_row):
        if t > 0:
            L = psi(L, h)
        L = L + dt
    return L


def _neg_ll_normative(params, d_matrix, y):
    a, b, h = params
    DV = np.array([_normative_dv(row, h) for row in d_matrix])
    linear = a + b * DV
    p = np.clip(expit(linear), 1e-10, 1 - 1e-10)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_normative(d_matrix, y, init=(0, 0.01, 0.1)):
    """
    Returns dict: params=[a, b, h], ll, n, k
    """
    bounds = [(None, None), (None, None), (1e-6, 0.5)]
    result = minimize(_neg_ll_normative, init, args=(d_matrix, y),
                       method='L-BFGS-B', bounds=bounds)

    return {'params': result.x, 'll': -result.fun, 'n': len(y), 'k': 3}


def predict_proba_normative(params, d_matrix):
    """P(choose left) for arbitrary d_matrix, given already-fitted params=[a, b, h]."""
    a, b, h = params
    DV = np.array([_normative_dv(row, h) for row in d_matrix])
    return expit(a + b * DV)


# ---------------------------------------------------------------------------
# Accuracy: how well does each fitted model reproduce the SUBJECT's actual
# choices (not the correct answer — pass high_side_final-derived y instead
# of choice if you want task accuracy rather than model-fit accuracy).
# ---------------------------------------------------------------------------

def _predicted_choice(a, b, DV):
    p = expit(a + b * DV)
    return (p > 0.5).astype(int)


def accuracy_perfect(params, d_matrix, y):
    a, b = params
    DV = d_matrix.sum(axis=1)
    pred = _predicted_choice(a, b, DV)
    return np.mean(pred == y)


def accuracy_leaky(params, d_matrix, y):
    a, b, lam = params
    n_frames = d_matrix.shape[1]
    weights = lam ** (n_frames - 1 - np.arange(n_frames))
    DV = d_matrix @ weights
    pred = _predicted_choice(a, b, DV)
    return np.mean(pred == y)


def accuracy_confirmation(params, d_matrix, y):
    a, b1, b2 = params
    X1 = d_matrix.sum(axis=1)
    X2 = np.array([_streak_regressor(row) for row in d_matrix])
    p = expit(a + b1 * X1 + b2 * X2)
    pred = (p > 0.5).astype(int)
    return np.mean(pred == y)


def accuracy_normative(params, d_matrix, y):
    a, b, h = params
    DV = np.array([_normative_dv(row, h) for row in d_matrix])
    pred = _predicted_choice(a, b, DV)
    return np.mean(pred == y)