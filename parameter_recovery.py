import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import pearsonr
from sys import path

# add the folder containing 01_integrators.py to the path
path.insert(0, '/Users/eliezerzucker/Documents/Glickman_Lab/lab_code')  # update this to your actual folder path

from integrators import (
    perfect_integrator,
    leaky_integrator,
    confirmation_weighted_integrator,
    bayesian_integrator_with_noise,
    MU_HIGH, MU_LOW, SIGMA_M, N_FRAMES, K
)

def generate_trials(n_trials, h, n_frames=N_FRAMES, mu_high=MU_HIGH, mu_low=MU_LOW, sigma_m=SIGMA_M):
    """
    Generates stimulus sequences for n_trials trials.
    Returns d_array of shape (n_trials, n_frames) and
    correct_side of shape (n_trials,) — 1 if Left is correct, 0 if Right.

    n_trials : int
    h        : float — swap probability per frame
    """
    d_array      = np.zeros((n_trials, n_frames))
    correct_side = np.zeros(n_trials, dtype=int)

    for i in range(n_trials):
        left_is_high  = True
        samples_left  = np.zeros(n_frames)
        samples_right = np.zeros(n_frames)

        for t in range(n_frames):
            if t > 0 and np.random.rand() < h:
                left_is_high = not left_is_high

            mu_left  = mu_high if left_is_high else mu_low
            mu_right = mu_low  if left_is_high else mu_high

            samples_left[t]  = np.random.normal(mu_left,  sigma_m)
            samples_right[t] = np.random.normal(mu_right, sigma_m)

        d_array[i]      = samples_left - samples_right
        correct_side[i] = 1 if left_is_high else 0

    return d_array, correct_side

def simulate_choices(d_array, integrator_fn, params):
    """
    Runs integrator_fn on each trial in d_array and returns binary choices.
    
    d_array       : array of shape (n_trials, n_frames)
    integrator_fn : one of perfect_integrator, leaky_integrator, etc.
    params        : dict of parameters for the integrator e.g. {"sigma": 10}
    returns       : array of shape (n_trials,) — 1 if chose Left, 0 if chose Right
    """
    n_trials, n_frames = d_array.shape
    choices = np.zeros(n_trials, dtype=int)

    for i in range(n_trials):
        epsilon = np.random.normal(0, 1, n_frames)
        L       = integrator_fn(d_array[i], epsilon, **params)
        choices[i] = 1 if L > 0 else 0

    return choices

def compute_nll_perfect(params, d_array, choices):
    """
    Negative log likelihood for the perfect integrator.

    Rather than running the stochastic integrator, we use the analytical
    Gaussian form of L to compute P(Left) directly on each trial.

    L ~ N(L_clean, sigma_L^2) where:
        L_clean = sum of d(t) across frames        — noise free evidence
        sigma_L = sqrt(2 * sigma^2 * n_frames)     — accumulated noise std

    P(chose Left) = Phi(L_clean / sigma_L)
    where Phi is the standard normal CDF.

    params   : array [sigma]
    d_array  : array of shape (n_trials, n_frames)
    choices  : array of shape (n_trials,) — 1 if chose Left, 0 if chose Right
    returns  : scalar NLL
    """
    sigma = params[0]

    n_trials, n_frames = d_array.shape

    # noise free evidence sum for each trial
    L_clean = d_array.sum(axis=1)                     # shape (n_trials,)

    # total accumulated noise std
    sigma_L = np.sqrt(2 * sigma**2 * n_frames)

    # guard against sigma = 0 — fully deterministic, CDF undefined
    if sigma_L == 0:
        p_left = (L_clean > 0).astype(float)
    else:
        p_left = norm.cdf(L_clean / sigma_L)          # shape (n_trials,)

    # probability assigned to the actual choice on each trial
    # if chose Left  (choice=1): probability is p_left
    # if chose Right (choice=0): probability is 1 - p_left
    p_choice = np.where(choices == 1, p_left, 1 - p_left)

    # clip to avoid log(0)
    p_choice = np.clip(p_choice, 1e-12, 1 - 1e-12)

    nll = -np.sum(np.log(p_choice))

    return nll

def fit_model_perfect(d_array, choices, n_restarts=3):
    """
    Fits the perfect integrator to data by minimizing NLL.
    Runs from n_restarts random starting points and returns
    the best result across all runs.

    d_array    : array of shape (n_trials, n_frames)
    choices    : array of shape (n_trials,) — 1 if chose Left, 0 if chose Right
    n_restarts : int — number of random starting points
    returns    : best_sigma, best_nll
    """
    bounds = [(0.5, 50)]   # sigma cannot be 0 — likelihood degenerates

    best_nll    = np.inf
    best_params = None

    for _ in range(n_restarts):

        # random starting point within bounds
        sigma_init    = np.random.uniform(0.5, 50)
        initial_guess = np.array([sigma_init])

        result = minimize(
            fun     = compute_nll_perfect,
            x0      = initial_guess,
            args    = (d_array, choices),
            method  = 'L-BFGS-B',
            bounds  = bounds
        )

        if result.fun < best_nll:
            best_nll    = result.fun
            best_params = result.x

    return best_params[0], best_nll   # sigma_recovered, nll

def run_recovery_perfect(n_subjects=100, n_trials=200, h=0.1):
    """
    For each synthetic subject:
        1. Draw a true sigma from a truncated Gaussian
        2. Generate trials
        3. Simulate choices using the perfect integrator
        4. Fit the model to recover sigma
        5. Save true vs recovered

    n_subjects : int — number of synthetic subjects
    n_trials   : int — trials per subject
    h          : float — swap probability used to generate trials
    returns    : dataframe with columns [sigma_true, sigma_recovered, nll]
    """
    results = []

    for i in range(n_subjects):

        # --- step 1: draw true sigma ---
        # sample from N(10, 5^2) truncated to [0.5, 50]
        while True:
            sigma_true = np.random.normal(10, 5)
            if 0.5 <= sigma_true <= 50:
                break

        # --- step 2: generate trials ---
        d_array, correct_side = generate_trials(n_trials, h)

        # --- step 3: simulate choices ---
        choices = simulate_choices(
            d_array       = d_array,
            integrator_fn = perfect_integrator,
            params        = {"sigma": sigma_true}
        )

        # --- step 4: fit model ---
        sigma_recovered, nll = fit_model_perfect(d_array, choices)

        # --- step 5: save ---
        results.append({
            "sigma_true":      sigma_true,
            "sigma_recovered": sigma_recovered,
            "nll":             nll
        })

        if (i + 1) % 10 == 0:
            print(f"  subject {i+1}/{n_subjects} done")

    return pd.DataFrame(results)

def compute_nll_leaky(params, d_array, choices, n_frames=N_FRAMES):
    """
    Negative log likelihood for the leaky integrator.

    L_clean is a weighted sum of evidence where earlier frames
    are down-weighted by the leak:
        L_clean = Σ_t (1-k)^(n_frames-1-t) * d(t)

    sigma_L is the std of L due to accumulated noise, also decayed:
        sigma_L² = 2*sigma² * Σ_t (1-k)^(2*(n_frames-1-t))

    params  : array [sigma, k]
    d_array : array of shape (n_trials, n_frames)
    choices : array of shape (n_trials,) — 1 if chose Left, 0 if chose Right
    returns : scalar NLL
    """
    sigma, k = params

    # decay weights for each frame — most recent frame has weight 1
    # frame 0 (oldest) has weight (1-k)^(n_frames-1)
    t       = np.arange(n_frames)
    weights = (1 - k) ** (n_frames - 1 - t)      # shape (n_frames,)

    # L_clean — weighted sum of evidence, different for each trial
    L_clean = d_array @ weights                   # shape (n_trials,)

    # sigma_L — same for every trial, depends only on sigma and k
    sigma_L = np.sqrt(2 * sigma**2 * np.sum(weights**2))

    if sigma_L == 0:
        p_left = (L_clean > 0).astype(float)
    else:
        p_left = norm.cdf(L_clean / sigma_L)

    p_choice = np.where(choices == 1, p_left, 1 - p_left)
    p_choice = np.clip(p_choice, 1e-12, 1 - 1e-12)

    return -np.sum(np.log(p_choice))

def fit_model_leaky(d_array, choices, n_restarts=3):
    """
    Fits the leaky integrator to data by minimizing NLL.
    Runs from n_restarts random starting points and returns
    the best result across all runs.

    d_array    : array of shape (n_trials, n_frames)
    choices    : array of shape (n_trials,) — 1 if chose Left, 0 if chose Right
    n_restarts : int — number of random starting points
    returns    : best_sigma, best_k, best_nll
    """
    bounds = [
        (0.5, 50),   # sigma — same reasoning as perfect integrator
        (0.0, 0.99)  # k — 0 means no leak, 1 means full leak (memory of 1 frame)
    ]

    best_nll    = np.inf
    best_params = None

    for _ in range(n_restarts):

        # random starting points within bounds
        sigma_init = np.random.uniform(0.5, 50)
        k_init     = np.random.uniform(0.0, 0.99)
        initial_guess = np.array([sigma_init, k_init])

        result = minimize(
            fun    = compute_nll_leaky,
            x0     = initial_guess,
            args   = (d_array, choices),
            method = 'L-BFGS-B',
            bounds = bounds
        )

        if result.fun < best_nll:
            best_nll    = result.fun
            best_params = result.x

    best_sigma, best_k = best_params
    return best_sigma, best_k, best_nll

def run_recovery_leaky(n_subjects=100, n_trials=200, h=0.1):
    """
    For each synthetic subject:
        1. Draw true sigma and k from truncated Gaussians
        2. Generate trials
        3. Simulate choices using the leaky integrator
        4. Fit the model to recover sigma and k
        5. Save true vs recovered

    n_subjects : int — number of synthetic subjects
    n_trials   : int — trials per subject
    h          : float — swap probability used to generate trials
    returns    : dataframe with columns [sigma_true, sigma_recovered,
                                         k_true, k_recovered, nll]
    """
    results = []

    for i in range(n_subjects):

        # --- step 1: draw true parameters ---
        # sigma ~ N(10, 5^2) truncated to [0.5, 50]
        while True:
            sigma_true = np.random.normal(10, 5)
            if 0.5 <= sigma_true <= 50:
                break

        # k ~ N(0.2, 0.1^2) truncated to [0.0, 0.99]
        while True:
            k_true = np.random.normal(0.2, 0.1)
            if 0.0 <= k_true <= 0.99:
                break

        # --- step 2: generate trials ---
        d_array, correct_side = generate_trials(n_trials, h)

        # --- step 3: simulate choices ---
        choices = simulate_choices(
            d_array       = d_array,
            integrator_fn = leaky_integrator,
            params        = {"sigma": sigma_true, "k": k_true}
        )

        # --- step 4: fit model ---
        sigma_recovered, k_recovered, nll = fit_model_leaky(d_array, choices)

        # --- step 5: save ---
        results.append({
            "sigma_true":      sigma_true,
            "sigma_recovered": sigma_recovered,
            "k_true":          k_true,
            "k_recovered":     k_recovered,
            "nll":             nll
        })

        if (i + 1) % 10 == 0:
            print(f"  subject {i+1}/{n_subjects} done")

    return pd.DataFrame(results)

def compute_nll_confirmation(params, d_array, choices, n_frames=N_FRAMES):
    """
    Negative log likelihood for the confirmation weighted integrator.

    Because confirmation weights depend on the sign pattern of each trial,
    we compute L_clean and sigma_L separately for each trial.

    For each trial:
        1. Compute confirmation weight per frame (w if confirmed, 1 if not)
        2. Combine with leak decay: effective_weight(t) = conf_weight(t) * (1-k)^(n_frames-1-t)
        3. L_clean = sum of effective_weight(t) * d(t)
        4. sigma_L  = sqrt(2 * sigma^2 * sum of effective_weight(t)^2)
        5. P(Left)  = Phi(L_clean / sigma_L)

    params  : array [sigma, k, w]
    d_array : array of shape (n_trials, n_frames)
    choices : array of shape (n_trials,) — 1 if chose Left, 0 if chose Right
    returns : scalar NLL
    """
    sigma, k, w = params

    n_trials = d_array.shape[0]

    # precompute leak decay weights — same for every trial
    t           = np.arange(n_frames)
    leak_decay  = (1 - k) ** (n_frames - 1 - t)   # shape (n_frames,)

    p_left = np.zeros(n_trials)

    for i in range(n_trials):
        d = d_array[i]

        # --- step 1: confirmation weights ---
        conf_weights = np.ones(n_frames)
        for t in range(n_frames - 1):
            # frame t gets upweighted if frame t+1 has same sign
            if np.sign(d[t + 1]) == np.sign(d[t]):
                conf_weights[t] = w
        # last frame always gets weight 1 — no confirmation possible

        # --- step 2: effective weights ---
        eff_weights = conf_weights * leak_decay

        # --- step 3: L_clean ---
        L_clean = np.sum(eff_weights * d)

        # --- step 4: sigma_L ---
        sigma_L = np.sqrt(2 * sigma**2 * np.sum(eff_weights**2))

        # --- step 5: P(Left) ---
        if sigma_L == 0:
            p_left[i] = 1.0 if L_clean > 0 else 0.0
        else:
            p_left[i] = norm.cdf(L_clean / sigma_L)

    p_choice = np.where(choices == 1, p_left, 1 - p_left)
    p_choice = np.clip(p_choice, 1e-12, 1 - 1e-12)

    return -np.sum(np.log(p_choice))


def fit_model_confirmation(d_array, choices, n_restarts=3):
    """
    Fits the confirmation weighted integrator to data by minimizing NLL.
    Runs from n_restarts random starting points and returns
    the best result across all runs.

    d_array    : array of shape (n_trials, n_frames)
    choices    : array of shape (n_trials,) — 1 if chose Left, 0 if chose Right
    n_restarts : int — number of random starting points
    returns    : best_sigma, best_k, best_w, best_nll
    """
    bounds = [
        (0.5, 50),    # sigma
        (0.0, 0.99),  # k
        (1.0, 5.0)    # w — 1 means no upweighting, 5 is a generous upper limit
    ]

    best_nll    = np.inf
    best_params = None

    for _ in range(n_restarts):

        sigma_init = np.random.uniform(0.5, 50)
        k_init     = np.random.uniform(0.0, 0.99)
        w_init     = np.random.uniform(1.0, 5.0)
        initial_guess = np.array([sigma_init, k_init, w_init])

        result = minimize(
            fun    = compute_nll_confirmation,
            x0     = initial_guess,
            args   = (d_array, choices),
            method = 'L-BFGS-B',
            bounds = bounds
        )

        if result.fun < best_nll:
            best_nll    = result.fun
            best_params = result.x

    best_sigma, best_k, best_w = best_params
    return best_sigma, best_k, best_w, best_nll


def run_recovery_confirmation(n_subjects=100, n_trials=200, h=0.1):
    """
    For each synthetic subject:
        1. Draw true sigma, k, w from truncated distributions
        2. Generate trials
        3. Simulate choices using the confirmation weighted integrator
        4. Fit the model to recover sigma, k, w
        5. Save true vs recovered

    n_subjects : int — number of synthetic subjects
    n_trials   : int — trials per subject
    h          : float — swap probability used to generate trials
    returns    : dataframe with columns [sigma_true, sigma_recovered,
                                         k_true, k_recovered,
                                         w_true, w_recovered, nll]
    """
    results = []

    for i in range(n_subjects):

        # --- step 1: draw true parameters ---
        while True:
            sigma_true = np.random.normal(10, 5)
            if 0.5 <= sigma_true <= 50:
                break

        while True:
            k_true = np.random.normal(0.2, 0.1)
            if 0.0 <= k_true <= 0.99:
                break

        while True:
            w_true = np.random.normal(2.0, 0.5)
            if 1.0 <= w_true <= 5.0:
                break

        # --- step 2: generate trials ---
        d_array, correct_side = generate_trials(n_trials, h)

        # --- step 3: simulate choices ---
        choices = simulate_choices(
            d_array       = d_array,
            integrator_fn = confirmation_weighted_integrator,
            params        = {"sigma": sigma_true, "k": k_true, "w": w_true}
        )

        # --- step 4: fit model ---
        sigma_recovered, k_recovered, w_recovered, nll = fit_model_confirmation(
            d_array, choices
        )

        # --- step 5: save ---
        results.append({
            "sigma_true":      sigma_true,
            "sigma_recovered": sigma_recovered,
            "k_true":          k_true,
            "k_recovered":     k_recovered,
            "w_true":          w_true,
            "w_recovered":     w_recovered,
            "nll":             nll
        })

        if (i + 1) % 10 == 0:
            print(f"  subject {i+1}/{n_subjects} done")

    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# Bayesian integrator with noise — NLL, fit, and recovery
# ---------------------------------------------------------------------------

def compute_nll_bayes_noise(params, d_array, choices, h,
                            n_frames=N_FRAMES, mu_high=MU_HIGH,
                            mu_low=MU_LOW, sigma_m=SIGMA_M):
    """
    Negative log likelihood for the Bayesian integrator with noise.

    Because the noise enters in log-odds space after the Bayesian update,
    there is no closed-form expression for P(Left) — we instead use a
    Monte Carlo approximation:  for each trial we draw S noise samples,
    run the integrator S times, and estimate P(Left) as the fraction of
    runs that produced a left choice.

    params   : array [sigma]  — internal noise scale
    d_array  : array of shape (n_trials, n_frames)
    choices  : array of shape (n_trials,) — 1 if chose Left, 0 if chose Right
    h        : float — swap probability (fixed at the generating value)
    returns  : scalar NLL
    """
    sigma    = params[0]
    std_d    = np.sqrt(2 * sigma_m**2)
    mu_0     = mu_high - mu_low
    mu_1     = mu_low  - mu_high
    sigma_lo = sigma / std_d

    n_trials = d_array.shape[0]
    S        = 50   # MC samples per trial — trade-off between speed and accuracy

    p_left = np.zeros(n_trials)

    for i in range(n_trials):
        d = d_array[i]
        # draw S independent noise sequences
        epsilons = np.random.normal(0, 1, (S, n_frames))   # (S, n_frames)

        chose_left_count = 0
        for s in range(S):
            p = 0.5
            for t in range(n_frames):
                p_prior = p * (1 - h) + (1 - p) * h if t > 0 else p

                L0 = norm.pdf(d[t], loc=mu_0, scale=std_d)
                L1 = norm.pdf(d[t], loc=mu_1, scale=std_d)
                denom = p_prior * L0 + (1 - p_prior) * L1
                p_updated = (p_prior * L0 / denom) if denom > 0 else p_prior

                p_clipped = np.clip(p_updated, 1e-9, 1 - 1e-9)
                log_odds   = np.log(p_clipped / (1 - p_clipped))
                log_odds  += np.sqrt(2) * sigma_lo * epsilons[s, t]
                p          = 1.0 / (1.0 + np.exp(-log_odds))

            chose_left_count += int(p > 0.5)

        p_left[i] = chose_left_count / S

    p_choice = np.where(choices == 1, p_left, 1 - p_left)
    p_choice  = np.clip(p_choice, 1e-12, 1 - 1e-12)

    return -np.sum(np.log(p_choice))


def fit_model_bayes_noise(d_array, choices, h, n_restarts=3):
    """
    Fits the Bayesian integrator with noise to data by minimizing NLL.
    Runs from n_restarts random starting points and returns the best result.

    d_array    : array of shape (n_trials, n_frames)
    choices    : array of shape (n_trials,) — 1 if chose Left, 0 if chose Right
    h          : float — swap probability (fixed, not fitted)
    n_restarts : int — number of random starting points
    returns    : best_sigma, best_nll
    """
    bounds = [(0.5, 50)]   # sigma

    best_nll    = np.inf
    best_params = None

    for _ in range(n_restarts):
        sigma_init    = np.random.uniform(0.5, 50)
        initial_guess = np.array([sigma_init])

        result = minimize(
            fun     = compute_nll_bayes_noise,
            x0      = initial_guess,
            args    = (d_array, choices, h),
            method  = 'L-BFGS-B',
            bounds  = bounds
        )

        if result.fun < best_nll:
            best_nll    = result.fun
            best_params = result.x

    return best_params[0], best_nll   # sigma_recovered, nll


def run_recovery_bayes_noise(n_subjects=100, n_trials=200, h=0.1):
    """
    Parameter recovery for the Bayesian integrator with noise.

    For each synthetic subject:
        1. Draw a true sigma from a truncated Gaussian
        2. Generate trials
        3. Simulate choices using bayesian_integrator_with_noise
        4. Fit the model to recover sigma
        5. Save true vs recovered

    n_subjects : int — number of synthetic subjects
    n_trials   : int — trials per subject
    h          : float — swap probability used to generate and fit the model
    returns    : dataframe with columns [sigma_true, sigma_recovered, nll]
    """
    results = []

    for i in range(n_subjects):

        # --- step 1: draw true sigma ---
        while True:
            sigma_true = np.random.normal(10, 5)
            if 0.5 <= sigma_true <= 50:
                break

        # --- step 2: generate trials ---
        d_array, _ = generate_trials(n_trials, h)

        # --- step 3: simulate choices ---
        choices = simulate_choices(
            d_array       = d_array,
            integrator_fn = bayesian_integrator_with_noise,
            params        = {"sigma": sigma_true, "h": h}
        )

        # --- step 4: fit model ---
        sigma_recovered, nll = fit_model_bayes_noise(d_array, choices, h)

        # --- step 5: save ---
        results.append({
            "sigma_true":      sigma_true,
            "sigma_recovered": sigma_recovered,
            "nll":             nll
        })

        if (i + 1) % 10 == 0:
            print(f"  subject {i+1}/{n_subjects} done")

    return pd.DataFrame(results)
