"""
Sequence Choice Analysis Pipeline
=================================
This script analyzes behavioral data from a decision-making experiment where
participants choose between two sequences of numbers (A and B).

Trial Types:
- Different-mean trials: Sequences have different means (tests accuracy)
- Same-mean trials: Sequences have same mean but different variance (tests risk preference)

Models Fitted:
1. Simple: P(choose A) = logistic(a + b * mean_difference)
2. Full: P(choose A) = logistic(a + b1*diff1 + ... + b8*diff8) - frame-by-frame weights
3. Leaky: P(choose A) = logistic(a + b * (λ^7·diff1 + ... + λ^0·diff8)) - exponential decay
4. SI-Leaky: Hybrid model combining selective integration with temporal decay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit  # Logistic (sigmoid) function: 1 / (1 + exp(-x))


def load_and_prepare_data(filepath):
    """
    Load experimental data and compute derived variables for analysis.
    
    Args:
        filepath: Path to CSV file containing raw experimental data
        
    Returns:
        DataFrame with original data plus computed columns
    """
    # Load raw data from CSV
    df = pd.read_csv(filepath)
    
    # Define column names for the 8 frames in each sequence
    # Each sequence shows 8 numbers to the participant
    seqA_cols = [f'sequenceA{i}' for i in range(1, 9)]  # sequenceA1, sequenceA2, ..., sequenceA8
    seqB_cols = [f'sequenceB{i}' for i in range(1, 9)]  # sequenceB1, sequenceB2, ..., sequenceB8
    
    # Compute summary statistics for each sequence
    df['meanA'] = df[seqA_cols].mean(axis=1)  # Average value of sequence A
    df['meanB'] = df[seqB_cols].mean(axis=1)  # Average value of sequence B
    df['varA'] = df[seqA_cols].var(axis=1)    # Variance of sequence A (spread/risk)
    df['varB'] = df[seqB_cols].var(axis=1)    # Variance of sequence B (spread/risk)
    
    # Mean difference: positive = A is better, negative = B is better
    df['meanDiff'] = df['meanA'] - df['meanB']
    
    # Binary choice outcome: 1 = chose A, 0 = chose B
    # Original column 'sequenceChoiceLeft' indicates if left (A) was chosen
    df['choseA'] = df['sequenceChoiceLeft']
    
    # Compute frame-by-frame differences (A - B) for each of the 8 positions
    # These are used in the full model to assess temporal weighting
    for i in range(1, 9):
        df[f'diff{i}'] = df[f'sequenceA{i}'] - df[f'sequenceB{i}']
    
    return df


def split_by_condition(df):
    """
    Split data into two trial types for separate analyses.
    
    Different-mean trials: Test whether participants correctly identify 
                          which sequence has higher expected value
    Same-mean trials: Test risk preference (broader = higher variance = riskier)
    
    Args:
        df: Full dataset
        
    Returns:
        Tuple of (different_mean_trials, same_mean_trials) DataFrames
    """
    # Split based on condition flag
    # isDifferentMeanSameVariance == 1: means differ, variance same (accuracy test)
    # isDifferentMeanSameVariance == 0: means same, variance differs (risk preference test)
    df_diffMean = df[df['isDifferentMeanSameVariance'] == 1].copy()
    df_sameMean = df[df['isDifferentMeanSameVariance'] == 0].copy()
    
    # For different-mean trials: compute whether choice was "correct"
    # Correct = chose the sequence with higher mean
    df_diffMean['correct'] = (
        ((df_diffMean['choseA'] == 1) & (df_diffMean['meanA'] > df_diffMean['meanB'])) |  # Chose A when A better
        ((df_diffMean['choseA'] == 0) & (df_diffMean['meanA'] < df_diffMean['meanB']))    # Chose B when B better
    )
    
    # For same-mean trials: compute risk preference
    # "Broader" = higher variance = riskier option
    df_sameMean['AisBroader'] = df_sameMean['varA'] > df_sameMean['varB']  # Is A the riskier option?
    
    # Did participant choose the riskier (broader) option?
    df_sameMean['choseBroader'] = (
        ((df_sameMean['choseA'] == 1) & df_sameMean['AisBroader']) |   # Chose A when A is broader
        ((df_sameMean['choseA'] == 0) & ~df_sameMean['AisBroader'])    # Chose B when B is broader
    )
    
    return df_diffMean, df_sameMean


def compute_descriptives(df_diffMean, df_sameMean):
    """
    Compute basic behavioral metrics.
    
    Args:
        df_diffMean: Different-mean trials
        df_sameMean: Same-mean trials
        
    Returns:
        Tuple of (accuracy_percent, preference_for_broader_percent)
    """
    # Accuracy: proportion of correct choices in different-mean trials
    accuracy = df_diffMean['correct'].mean() * 100
    
    # Risk preference: proportion choosing the higher-variance option
    # >50% = risk-seeking, <50% = risk-averse, =50% = risk-neutral
    pref_broad = df_sameMean['choseBroader'].mean() * 100
    
    return accuracy, pref_broad


def neg_log_likelihood(params, X, y):
    """
    Compute negative log-likelihood for logistic regression.
    Used as the objective function for optimization.
    
    Logistic model: P(y=1) = sigmoid(X @ params)
    
    Args:
        params: Model parameters (intercept + coefficients)
        X: Design matrix (n_trials x n_features)
        y: Binary outcomes (0 or 1)
        
    Returns:
        Negative log-likelihood (scalar to minimize)
    """
    # Compute linear combination: intercept + weighted sum of predictors
    linear = X @ params
    
    # Apply logistic function to get probabilities
    p = expit(linear)
    
    # Clip probabilities to avoid log(0) errors
    p = np.clip(p, 1e-10, 1 - 1e-10)
    
    # Binary cross-entropy loss (negative log-likelihood)
    # LL = sum(y * log(p) + (1-y) * log(1-p))
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_logistic_simple(df_diffMean):
    """
    Fit the simple logistic model to each subject's data.
    
    Model: P(choose A) = logistic(a + b * meanDiff)
    
    This model assumes choices depend only on the overall mean difference,
    regardless of how that difference is distributed across frames.
    
    Args:
        df_diffMean: Different-mean trials
        
    Returns:
        Dict mapping subject ID to {params, log_likelihood, n_trials}
    """
    results = {}
    
    # Fit model separately for each subject
    for subj in df_diffMean['subject'].unique():
        # Get this subject's data
        df_subj = df_diffMean[df_diffMean['subject'] == subj]
        
        # Build design matrix: column of 1s (intercept) + mean difference
        X = np.column_stack([
            np.ones(len(df_subj)),           # Intercept term
            df_subj['meanDiff'].values       # Predictor: mean(A) - mean(B)
        ])
        
        # Outcome: did they choose A?
        y = df_subj['choseA'].values
        
        # Optimize: find params that minimize negative log-likelihood
        # Initial guess: intercept=0, slope=0.1
        result = minimize(
            neg_log_likelihood, 
            [0, 0.1],           # Starting values
            args=(X, y),        # Additional arguments to objective function
            method='L-BFGS-B'   # Quasi-Newton optimization
        )
        
        # Store results
        results[subj] = {
            'params': result.x,      # [intercept, slope]
            'll': -result.fun,       # Log-likelihood (negate the negative)
            'n': len(df_subj)        # Number of trials for BIC calculation
        }
    
    return results


def fit_logistic_full(df_diffMean):
    """
    Fit the full logistic model with frame-by-frame weights.
    
    Model: P(choose A) = logistic(a + b1*diff1 + b2*diff2 + ... + b8*diff8)
    
    This model allows each frame position to have a different weight,
    testing for primacy effects (early frames weighted more) or 
    recency effects (late frames weighted more).
    
    Args:
        df_diffMean: Different-mean trials
        
    Returns:
        Dict mapping subject ID to {params, log_likelihood, n_trials}
    """
    # Column names for frame-by-frame differences
    diff_cols = [f'diff{i}' for i in range(1, 9)]
    results = {}
    
    for subj in df_diffMean['subject'].unique():
        df_subj = df_diffMean[df_diffMean['subject'] == subj]
        
        # Design matrix: intercept + 8 frame differences
        X = np.column_stack([
            np.ones(len(df_subj)),        # Intercept
            df_subj[diff_cols].values     # 8 columns of frame differences
        ])
        
        y = df_subj['choseA'].values
        
        # Initialize all 9 parameters (1 intercept + 8 weights) to 0
        init_params = np.zeros(9)
        
        result = minimize(
            neg_log_likelihood, 
            init_params, 
            args=(X, y), 
            method='L-BFGS-B'
        )
        
        results[subj] = {
            'params': result.x,   # [intercept, b1, b2, ..., b8]
            'll': -result.fun,
            'n': len(df_subj)
        }
    
    return results


def neg_log_likelihood_leaky(params, diff_matrix, y):
    """
    Negative log-likelihood for leaky integration model.
    
    Model: P(choose A) = logistic(a + b * weighted_sum)
    where weighted_sum = λ^7·diff1 + λ^6·diff2 + ... + λ^0·diff8
    
    This implements a recency-weighted integration where later frames
    have more influence (when λ < 1). The "leak" refers to information
    from earlier frames being progressively forgotten/discounted.
    
    Args:
        params: [a, b, λ] - intercept, slope, leak parameter
        diff_matrix: (n_trials, 8) array of frame differences (A - B)
        y: Binary outcomes (1 = chose A, 0 = chose B)
        
    Returns:
        Negative log-likelihood (scalar to minimize)
    """
    # Unpack parameters
    a = params[0]    # Intercept (bias toward A or B)
    b = params[1]    # Slope (sensitivity to evidence)
    lam = params[2]  # Leak parameter (0 to 1)
    
    # Compute exponential weights for each frame position
    # Frame 1 (earliest): weight = λ^7 (smallest if λ < 1)
    # Frame 2: weight = λ^6
    # ...
    # Frame 8 (latest): weight = λ^0 = 1 (always full weight)
    weights = np.array([lam ** (7 - i) for i in range(8)])  # [λ^7, λ^6, ..., λ^1, λ^0]
    
    # Compute weighted sum of evidence for each trial
    # diff_matrix is (n_trials, 8), weights is (8,)
    # Result is (n_trials,) - one weighted evidence value per trial
    weighted_sum = diff_matrix @ weights
    
    # Linear predictor: intercept + slope * weighted_evidence
    linear = a + b * weighted_sum
    
    # Convert to probability using logistic function
    p = expit(linear)
    
    # Clip probabilities to avoid log(0) numerical errors
    p = np.clip(p, 1e-10, 1 - 1e-10)
    
    # Compute negative log-likelihood (binary cross-entropy)
    # We minimize this, so negative of the log-likelihood
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def neg_log_likelihood_si_leaky(params, seqA_matrix, seqB_matrix, y):
    """
    Negative log-likelihood for hybrid SI-Leaky model.
    
    Combines selective integration (value-based weighting within frames)
    with leaky integration (temporal weighting across frames).
    
    SI weighting operates on individual values within each frame:
    - Higher value gets weight w = sigmoid(β * |A_i - B_i|)
    - Lower value gets weight (1 - w)
    - This simulates goal-consistent attentional selection
    
    Model: P(choose A) = logistic(a + b * weighted_sum)
    where weighted_sum combines SI and temporal weighting.
    
    Args:
        params: [a, b, β, λ] - intercept, slope, SI selectivity, leak parameter
        seqA_matrix: (n_trials, 8) array of sequence A values
        seqB_matrix: (n_trials, 8) array of sequence B values
        y: Binary outcomes (1 = chose A, 0 = chose B)
        
    Returns:
        Negative log-likelihood (scalar to minimize)
    """
    # Unpack parameters
    a = params[0]    # Intercept (bias toward A or B)
    b = params[1]    # Slope (sensitivity to evidence)
    beta = params[2] # SI selectivity (value-dependent attention)
    lam = params[3]  # Leak parameter (0 to 1) for temporal decay
    
    # Compute differences to determine which value is higher in each frame
    diff_matrix = seqA_matrix - seqB_matrix  # (n_trials, 8)
    abs_diff = np.abs(diff_matrix)
    
    # SI weighting: higher value gets more weight
    # w_higher = sigmoid(β * |difference|)
    w_higher = expit(beta * abs_diff)  # (n_trials, 8)
    w_lower = 1 - w_higher
    
    # Apply weights to actual values based on which is higher
    # When A > B: weighted_diff = w_higher * A - w_lower * B
    # When B > A: weighted_diff = w_lower * A - w_higher * B
    A_is_higher = (diff_matrix > 0).astype(float)  # 1 when A > B, 0 otherwise
    
    # Weight for A: w_higher when A > B, w_lower when B > A
    w_A = w_higher * A_is_higher + w_lower * (1 - A_is_higher)
    # Weight for B: w_lower when A > B, w_higher when B > A  
    w_B = w_lower * A_is_higher + w_higher * (1 - A_is_higher)
    
    # Compute weighted differences
    si_weighted_diff = w_A * seqA_matrix - w_B * seqB_matrix  # (n_trials, 8)
    
    # Apply temporal weights (recency effect)
    temporal_weights = np.array([lam ** (7 - i) for i in range(8)])  # [λ^7, ..., λ^0]
    
    # Combine SI and temporal weighting
    weighted_sum = si_weighted_diff @ temporal_weights  # (n_trials,)
    
    # Linear predictor
    linear = a + b * weighted_sum
    
    # Convert to probability using logistic function
    p = expit(linear)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    
    # Compute negative log-likelihood
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_logistic_leaky(df_diffMean):
    """
    Fit the leaky integration model to each subject's data.
    
    Model: P(choose A) = logistic(a + b * (λ^7·diff1 + λ^6·diff2 + ... + λ^0·diff8))
    
    This model assumes exponential discounting of earlier frames,
    with λ controlling the decay rate:
    - λ = 1: all frames weighted equally (equivalent to simple model)
    - λ < 1: recency effect (later frames weighted more heavily)
    - λ → 0: extreme recency (only the last frame matters)
    
    The model has 3 parameters vs 9 in the full model, making it more
    parsimonious while still capturing temporal weighting patterns.
    
    Args:
        df_diffMean: Different-mean trials DataFrame
        
    Returns:
        Dict mapping subject ID to {params: [a, b, λ], ll: log_likelihood, n: n_trials}
    """
    # Column names for the 8 frame differences
    diff_cols = [f'diff{i}' for i in range(1, 9)]
    results = {}
    
    # Fit model separately for each subject
    for subj in df_diffMean['subject'].unique():
        # Get this subject's data
        df_subj = df_diffMean[df_diffMean['subject'] == subj]
        
        # Extract frame differences as matrix
        # Shape: (n_trials, 8) where each row is a trial, each column is a frame
        diff_matrix = df_subj[diff_cols].values
        
        # Outcome: did they choose A? (1 = yes, 0 = no)
        y = df_subj['choseA'].values
        
        # Initial parameter guesses
        # a = 0: no initial bias
        # b = 0.1: moderate sensitivity to evidence
        # λ = 0.9: slight recency effect as starting point
        init_params = [0, 0.1, 0.9]
        
        # Parameter bounds
        # a: unbounded (can have any bias)
        # b: unbounded (can have positive or negative sensitivity)
        # λ: bounded (0, 1] - must be positive, at most 1
        bounds = [(None, None), (None, None), (1e-6, 1.0)]
        
        # Optimize: find parameters that minimize negative log-likelihood
        result = minimize(
            neg_log_likelihood_leaky,  # Objective function
            init_params,                # Starting values
            args=(diff_matrix, y),      # Additional arguments
            method='L-BFGS-B',          # Bounded quasi-Newton method
            bounds=bounds               # Parameter constraints
        )
        
        # Store results for this subject
        results[subj] = {
            'params': result.x,   # Fitted parameters: [a, b, λ]
            'll': -result.fun,    # Log-likelihood (negate the negative)
            'n': len(df_subj)     # Number of trials (for BIC calculation)
        }
    
    return results


def fit_logistic_si_leaky(df_all):
    """
    Fit the hybrid SI-Leaky integration model to each subject's data.
    
    Model: P(choose A) = logistic(a + b * weighted_sum)
    where weighted_sum combines:
    - SI weighting: higher values in each frame get more weight
    - Temporal weighting: λ^(8-i) creates recency effect
    
    This hybrid model tests whether both value-based (SI) and temporal (leaky)
    components contribute to choice behavior.
    
    The model has 4 parameters:
    - a: intercept/bias
    - b: sensitivity to evidence
    - β: SI selectivity (typically small, ~0.1)
    - λ: temporal decay (0 to 1)
    
    Args:
        df_all: All trials DataFrame
        
    Returns:
        Dict mapping subject ID to {params: [a, b, β, λ], ll: log_likelihood, n: n_trials}
    """
    # Column names for the 8 frames in each sequence
    seqA_cols = [f'sequenceA{i}' for i in range(1, 9)]
    seqB_cols = [f'sequenceB{i}' for i in range(1, 9)]
    results = {}
    
    # Fit model separately for each subject
    for subj in df_all['subject'].unique():
        # Get this subject's data
        df_subj = df_all[df_all['subject'] == subj]
        
        # Extract actual sequence values as matrices
        seqA_matrix = df_subj[seqA_cols].values  # (n_trials, 8)
        seqB_matrix = df_subj[seqB_cols].values  # (n_trials, 8)
        
        # Outcome: did they choose A?
        y = df_subj['choseA'].values
        
        # Initial parameter guesses
        # a = 0: no initial bias
        # b = 0.1: moderate sensitivity
        # β = 0.1: small SI effect (as suggested)
        # λ = 0.9: slight recency effect
        init_params = [0, 0.1, 0.1, 0.9]
        
        # Parameter bounds
        # a, b: unbounded
        # β: positive (SI amplification)
        # λ: (0, 1] for temporal decay
        bounds = [(None, None), (None, None), (1e-6, 10.0), (1e-6, 1.0)]
        
        # Optimize
        result = minimize(
            neg_log_likelihood_si_leaky,
            init_params,
            args=(seqA_matrix, seqB_matrix, y),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        # Store results
        results[subj] = {
            'params': result.x,   # [a, b, β, λ]
            'll': -result.fun,
            'n': len(df_subj)
        }
    
    return results


def compare_models(results_simple, results_full, results_leaky=None, results_si_leaky=None):
    """
    Compare models using information criteria.
    
    AIC (Akaike): -2*LL + 2*k  (penalizes complexity less)
    BIC (Bayesian): -2*LL + k*log(n)  (penalizes complexity more)
    
    Lower values = better model (balancing fit and parsimony)
    
    Args:
        results_simple: Output from fit_logistic_simple
        results_full: Output from fit_logistic_full
        results_leaky: Output from fit_logistic_leaky (optional)
        results_si_leaky: Output from fit_logistic_si_leaky (optional)
        
    Returns:
        Dict with AIC, BIC, and log-likelihood for all models
    """
    # Number of free parameters in each model
    k_simple = 2      # intercept + 1 slope
    k_full = 9        # intercept + 8 frame weights
    k_leaky = 3       # intercept + slope + lambda
    k_si_leaky = 4    # intercept + slope + beta + lambda
    
    # Initialize accumulators
    aic_simple = 0
    aic_full = 0
    aic_leaky = 0
    aic_si_leaky = 0
    bic_simple = 0
    bic_full = 0
    bic_leaky = 0
    bic_si_leaky = 0
    ll_simple = 0
    ll_full = 0
    ll_leaky = 0
    ll_si_leaky = 0
    
    # Compute fit statistics per subject, then sum
    # (This is appropriate when fitting separate models per subject)
    for subj in results_simple.keys():
        n_trials = results_simple[subj]['n']  # Number of trials for this subject
        ll_s = results_simple[subj]['ll']     # Log-likelihood, simple model
        ll_f = results_full[subj]['ll']       # Log-likelihood, full model
        
        # Accumulate total log-likelihood
        ll_simple += ll_s
        ll_full += ll_f
        
        # AIC: -2*LL + 2*k (per subject, then summed)
        aic_simple += -2 * ll_s + 2 * k_simple
        aic_full += -2 * ll_f + 2 * k_full
        
        # BIC: -2*LL + k*log(n) (n = trials for this subject)
        bic_simple += -2 * ll_s + k_simple * np.log(n_trials)
        bic_full += -2 * ll_f + k_full * np.log(n_trials)
        
        # Leaky model (if provided)
        if results_leaky is not None:
            ll_l = results_leaky[subj]['ll']
            ll_leaky += ll_l
            aic_leaky += -2 * ll_l + 2 * k_leaky
            bic_leaky += -2 * ll_l + k_leaky * np.log(n_trials)
        
        # SI-Leaky model (if provided)
        if results_si_leaky is not None:
            ll_sl = results_si_leaky[subj]['ll']
            ll_si_leaky += ll_sl
            aic_si_leaky += -2 * ll_sl + 2 * k_si_leaky
            bic_si_leaky += -2 * ll_sl + k_si_leaky * np.log(n_trials)
    
    # Summary statistics
    n_obs = sum(r['n'] for r in results_simple.values())  # Total trials across all subjects
    n_subjects = len(results_simple)
    
    result = {
        'aic_simple': aic_simple, 'aic_full': aic_full,
        'bic_simple': bic_simple, 'bic_full': bic_full,
        'll_simple': ll_simple, 'll_full': ll_full,
        'n_subjects': n_subjects, 'n_obs': n_obs
    }
    
    # Add leaky model results if available
    if results_leaky is not None:
        result.update({
            'aic_leaky': aic_leaky,
            'bic_leaky': bic_leaky,
            'll_leaky': ll_leaky
        })
    
    # Add SI-Leaky model results if available
    if results_si_leaky is not None:
        result.update({
            'aic_si_leaky': aic_si_leaky,
            'bic_si_leaky': bic_si_leaky,
            'll_si_leaky': ll_si_leaky
        })
    
    return result


def plot_individual_accuracy(df_diffMean, df_sameMean):
    """
    Plot individual subject performance, sorted from lowest to highest.
    Helps visualize individual differences and identify outliers.
    
    Left panel: Accuracy on different-mean trials
    Right panel: Risk preference on same-mean trials
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # === Left Panel: Accuracy ===
    # Compute accuracy for each subject
    acc_by_subj = df_diffMean.groupby('subject')['correct'].mean() * 100
    acc_sorted = acc_by_subj.sort_values()  # Sort for visualization
    
    ax1 = axes[0]
    # Plot individual subjects as connected dots
    ax1.plot(
        range(len(acc_sorted)),      # X: subject rank (0 to N-1)
        acc_sorted.values,           # Y: accuracy percentage
        'o-',                        # Circle markers connected by lines
        color='steelblue', 
        markersize=8, 
        markerfacecolor='steelblue', 
        markeredgecolor='black', 
        linewidth=1.5
    )
    # Reference lines
    ax1.axhline(50, color='red', linestyle='--', label='Chance')  # 50% = random guessing
    ax1.axhline(acc_sorted.mean(), color='green', linestyle='-', lw=2, 
                label=f'Mean = {acc_sorted.mean():.1f}%')
    
    # Formatting
    ax1.set_xlabel('Subject (sorted)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    ax1.set_title('Individual Accuracy (Different-Mean Trials)')
    ax1.legend()
    
    # === Right Panel: Risk Preference ===
    pref_by_subj = df_sameMean.groupby('subject')['choseBroader'].mean() * 100
    pref_sorted = pref_by_subj.sort_values()
    
    ax2 = axes[1]
    ax2.plot(
        range(len(pref_sorted)), 
        pref_sorted.values, 
        'o-', 
        color='coral',
        markersize=8, 
        markerfacecolor='coral', 
        markeredgecolor='black', 
        linewidth=1.5
    )
    ax2.axhline(50, color='red', linestyle='--', label='No preference')  # 50% = risk-neutral
    ax2.axhline(pref_sorted.mean(), color='green', linestyle='-', lw=2, 
                label=f'Mean = {pref_sorted.mean():.1f}%')
    
    ax2.set_xlabel('Subject (sorted)')
    ax2.set_ylabel('Preference for Broader (%)')
    ax2.set_ylim(0, 100)
    ax2.set_title('Individual Risk Preference (Same-Mean Trials)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def plot_descriptives(df_diffMean, df_sameMean):
    """
    Plot summary bar charts with error bars (SEM across subjects).
    
    Left panel: Mean accuracy with standard error
    Right panel: Mean preference for broader (riskier) option
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Compute per-subject means first (for proper SEM calculation)
    acc_by_subj = df_diffMean.groupby('subject')['correct'].mean() * 100
    accuracy = acc_by_subj.mean()                              # Grand mean
    acc_se = acc_by_subj.std() / np.sqrt(len(acc_by_subj))    # Standard error of mean
    
    pref_by_subj = df_sameMean.groupby('subject')['choseBroader'].mean() * 100
    pref_broad = pref_by_subj.mean()
    pref_se = pref_by_subj.std() / np.sqrt(len(pref_by_subj))
    
    # Sample sizes for titles
    n_diff = len(df_diffMean)  # Total different-mean trials
    n_same = len(df_sameMean)  # Total same-mean trials
    
    # === Left Panel: Accuracy Bar ===
    ax1 = axes[0]
    ax1.bar(
        ['Mean'],                    # Single bar
        [accuracy],                  # Height = mean accuracy
        yerr=[acc_se],              # Error bar = SEM
        color='steelblue', 
        edgecolor='black', 
        width=0.5, 
        capsize=5                    # Width of error bar caps
    )
    ax1.axhline(50, color='red', linestyle='--', label='Chance')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    ax1.set_title(f'Different-Mean Trials (N={n_diff})')
    # Add text label above bar
    ax1.text(0, accuracy + acc_se + 3, f'{accuracy:.1f}%', ha='center', fontweight='bold')
    ax1.legend()
    
    # === Right Panel: Risk Preference Bar ===
    ax2 = axes[1]
    ax2.bar(
        ['Mean'], 
        [pref_broad], 
        yerr=[pref_se], 
        color='coral', 
        edgecolor='black', 
        width=0.5, 
        capsize=5
    )
    ax2.axhline(50, color='red', linestyle='--', label='No preference')
    ax2.set_ylabel('Preference (%)')
    ax2.set_ylim(0, 100)
    ax2.set_title(f'Same-Mean Trials (N={n_same})')
    ax2.text(0, pref_broad + pref_se + 3, f'{pref_broad:.1f}%', ha='center', fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def plot_model_comparison(comparison):
    """
    Plot bar charts comparing models on log-likelihood, AIC, and BIC.
    
    This visualization helps assess which model best balances fit and parsimony:
    - Log-likelihood: Higher = better fit (but doesn't penalize complexity)
    - AIC: Lower = better (penalizes complexity mildly)
    - BIC: Lower = better (penalizes complexity more strongly)
    
    Model comparison logic:
    - If full model wins: temporal weighting matters, frames have different influence
    - If leaky wins: recency effect exists, but constrained exponential form is sufficient
    - If SI-Leaky wins: both value-based and temporal weighting matter
    - If simple wins: only overall mean difference matters, no temporal effects
    
    Args:
        comparison: Dict from compare_models() containing LL, AIC, BIC for all models
    """
    # Check which model results are available
    has_leaky = 'll_leaky' in comparison
    has_si_leaky = 'll_si_leaky' in comparison
    
    # Prepare data for plotting
    models = ['Simple\n(k=2)', 'Full\n(k=9)']
    ll_vals = [comparison['ll_simple'], comparison['ll_full']]
    aic_vals = [comparison['aic_simple'], comparison['aic_full']]
    bic_vals = [comparison['bic_simple'], comparison['bic_full']]
    colors = ['steelblue', 'teal']
    
    if has_leaky:
        models.append('Leaky\n(k=3)')
        ll_vals.append(comparison['ll_leaky'])
        aic_vals.append(comparison['aic_leaky'])
        bic_vals.append(comparison['bic_leaky'])
        colors.append('orange')
    
    if has_si_leaky:
        models.append('SI-Leaky\n(k=4)')
        ll_vals.append(comparison['ll_si_leaky'])
        aic_vals.append(comparison['aic_si_leaky'])
        bic_vals.append(comparison['bic_si_leaky'])
        colors.append('purple')
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # === Left Panel: Log-Likelihood ===
    # Higher is better (better fit to data)
    ax1 = axes[0]
    bars1 = ax1.bar(models, ll_vals, color=colors, edgecolor='black')
    
    # Highlight the best (highest) LL
    best_ll_idx = np.argmax(ll_vals)
    bars1[best_ll_idx].set_edgecolor('red')
    bars1[best_ll_idx].set_linewidth(3)
    
    ax1.set_ylabel('Log-Likelihood')
    ax1.set_title('Model Fit (higher = better)')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, ll_vals)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + abs(max(ll_vals))*0.01,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # === Middle Panel: AIC ===
    # Lower is better (balances fit and complexity)
    ax2 = axes[1]
    bars2 = ax2.bar(models, aic_vals, color=colors, edgecolor='black')
    
    # Highlight the best (lowest) AIC
    best_aic_idx = np.argmin(aic_vals)
    bars2[best_aic_idx].set_edgecolor('red')
    bars2[best_aic_idx].set_linewidth(3)
    
    ax2.set_ylabel('AIC')
    ax2.set_title('AIC (lower = better)')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, aic_vals)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(aic_vals)*0.01,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # === Right Panel: BIC ===
    # Lower is better (penalizes complexity more than AIC)
    ax3 = axes[2]
    bars3 = ax3.bar(models, bic_vals, color=colors, edgecolor='black')
    
    # Highlight the best (lowest) BIC
    best_bic_idx = np.argmin(bic_vals)
    bars3[best_bic_idx].set_edgecolor('red')
    bars3[best_bic_idx].set_linewidth(3)
    
    ax3.set_ylabel('BIC')
    ax3.set_title('BIC (lower = better)')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars3, bic_vals)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(bic_vals)*0.01,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary of winners
    model_names = ['Simple', 'Full']
    if has_leaky:
        model_names.append('Leaky')
    if has_si_leaky:
        model_names.append('SI-Leaky')
    
    print(f"\nModel comparison summary:")
    print(f"  Best fit (LL):  {model_names[best_ll_idx]}")
    print(f"  Best AIC:       {model_names[best_aic_idx]}")
    print(f"  Best BIC:       {model_names[best_bic_idx]}")


def plot_lambda_distribution(results_leaky, results_si_leaky=None):
    """
    Plot the distribution of leak (λ) parameters across subjects.
    
    Shows histogram and individual points to visualize how λ varies
    across participants. This helps assess:
    - Whether there's a consistent recency effect (λ < 1) across subjects
    - Individual differences in temporal weighting
    - Whether λ clusters near specific values
    
    Interpretation of λ:
    - λ = 1: All frames weighted equally (no leak)
    - λ < 1: Recency effect (later frames weighted more heavily)
    - λ → 0: Extreme recency (only last frame matters)
    """
    # Extract lambda (3rd parameter) from each subject's fitted model
    # results_leaky is a dict: {subject_id: {'params': [a, b, λ], 'll': ..., 'n': ...}}
    lambdas_leaky = np.array([r['params'][2] for r in results_leaky.values()])
    
    # If SI-Leaky results provided, extract those lambdas too (4th parameter)
    if results_si_leaky is not None:
        lambdas_si_leaky = np.array([r['params'][3] for r in results_si_leaky.values()])
    
    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # === Left Panel: Histogram ===
    # Shows the frequency distribution of λ values
    ax1 = axes[0]
    ax1.hist(
        lambdas_leaky, 
        bins=15,                    # Number of bins for histogram
        color='orange', 
        edgecolor='black', 
        alpha=0.7,                  # Slight transparency
        label='Leaky'
    )
    
    if results_si_leaky is not None:
        ax1.hist(
            lambdas_si_leaky,
            bins=15,
            color='purple',
            edgecolor='black',
            alpha=0.5,
            label='SI-Leaky'
        )
    
    # Add vertical line at mean λ for each model
    ax1.axvline(
        lambdas_leaky.mean(), 
        color='orange', 
        linestyle='-', 
        lw=2, 
        label=f'Leaky Mean = {lambdas_leaky.mean():.3f}'
    )
    
    if results_si_leaky is not None:
        ax1.axvline(
            lambdas_si_leaky.mean(),
            color='purple',
            linestyle='-',
            lw=2,
            label=f'SI-Leaky Mean = {lambdas_si_leaky.mean():.3f}'
        )
    
    # Add reference line at λ=1 (equal weighting / no leak)
    ax1.axvline(
        1.0, 
        color='gray', 
        linestyle='--', 
        lw=1.5, 
        label='λ=1 (equal weights)'
    )
    
    # Axis labels and formatting
    ax1.set_xlabel('λ (leak parameter)')
    ax1.set_ylabel('Number of subjects')
    ax1.set_title('Distribution of Leak Parameter')
    ax1.set_xlim(0, 1.05)  # λ is bounded [0, 1], add small buffer
    ax1.legend(fontsize=8)
    
    # === Right Panel: Individual points (sorted) ===
    # Shows each subject's λ value, sorted from lowest to highest
    # Useful for identifying outliers and seeing the full range
    ax2 = axes[1]
    
    # Sort λ values for cleaner visualization
    lambdas_leaky_sorted = np.sort(lambdas_leaky)
    
    # Plot as connected dots (same style as accuracy plots)
    ax2.plot(
        range(len(lambdas_leaky_sorted)),  # X: subject rank (0 to N-1)
        lambdas_leaky_sorted,               # Y: sorted λ value
        'o-',                               # Circle markers connected by lines
        color='orange', 
        markersize=8, 
        markerfacecolor='orange', 
        markeredgecolor='black', 
        linewidth=1.5,
        label='Leaky'
    )
    
    if results_si_leaky is not None:
        lambdas_si_leaky_sorted = np.sort(lambdas_si_leaky)
        ax2.plot(
            range(len(lambdas_si_leaky_sorted)),
            lambdas_si_leaky_sorted,
            's-',  # Square markers
            color='purple',
            markersize=8,
            markerfacecolor='purple',
            markeredgecolor='black',
            linewidth=1.5,
            label='SI-Leaky'
        )
    
    # Reference lines
    ax2.axhline(
        lambdas_leaky.mean(), 
        color='orange', 
        linestyle='-', 
        lw=2, 
        label=f'Leaky Mean = {lambdas_leaky.mean():.3f}'
    )
    
    if results_si_leaky is not None:
        ax2.axhline(
            lambdas_si_leaky.mean(),
            color='purple',
            linestyle='-',
            lw=2,
            label=f'SI-Leaky Mean = {lambdas_si_leaky.mean():.3f}'
        )
    
    ax2.axhline(
        1.0, 
        color='gray', 
        linestyle='--', 
        lw=1.5, 
        label='λ=1 (equal weights)'
    )
    
    # Axis labels and formatting
    ax2.set_xlabel('Subject (sorted)')
    ax2.set_ylabel('λ (leak parameter)')
    ax2.set_title('Individual Leak Parameters')
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics to console
    print(f"\nLeaky model - Lambda summary: M = {lambdas_leaky.mean():.3f}, SD = {lambdas_leaky.std():.3f}, "
          f"range = [{lambdas_leaky.min():.3f}, {lambdas_leaky.max():.3f}]")
    
    if results_si_leaky is not None:
        print(f"SI-Leaky model - Lambda summary: M = {lambdas_si_leaky.mean():.3f}, SD = {lambdas_si_leaky.std():.3f}, "
              f"range = [{lambdas_si_leaky.min():.3f}, {lambdas_si_leaky.max():.3f}]")


def plot_logistic_fits(df_all, results_simple, results_full, results_leaky=None, results_si_leaky=None):
    """
    Visualize the fitted logistic models.
    
    Left panel: Psychometric curves from simple model (choice probability vs mean difference)
    Right panel: Frame weights from full model (tests primacy/recency effects)
                 + implied weights from leaky and SI-leaky models (if provided)
    
    Args:
        df_all: All trials (used for computing average SI weights)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # === Left Panel: Psychometric Curves ===
    ax1 = axes[0]
    
    # X-axis: range of possible mean differences
    x_range = np.linspace(-25, 25, 100)
    
    # Extract parameters from all subjects
    all_params = np.array([r['params'] for r in results_simple.values()])
    mean_a, mean_b = all_params.mean(axis=0)  # Average intercept and slope
    
    # Plot each subject's psychometric curve (gray, transparent)
    for params in all_params:
        # P(choose A) = sigmoid(intercept + slope * meanDiff)
        y_pred = expit(params[0] + params[1] * x_range)
        ax1.plot(x_range, y_pred, 'gray', alpha=0.3, lw=1)
    
    # Plot mean curve (red, bold)
    y_pred_mean = expit(mean_a + mean_b * x_range)
    ax1.plot(x_range, y_pred_mean, 'r-', lw=2, label=f'Mean fit (b={mean_b:.3f})')
    
    # Reference lines
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)  # 50% = indifference
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)    # 0 = equal means
    
    ax1.set_xlabel('Mean Difference (A - B)')
    ax1.set_ylabel('P(Choose A)')
    ax1.set_title('Model 1: P(A) = logistic(a + b·meanDiff)')
    ax1.legend()
    ax1.set_xlim(-25, 25)
    ax1.set_ylim(0, 1)
    
    # === Right Panel: Frame Weights ===
    ax2 = axes[1]
    
    # Extract beta weights (excluding intercept) from all subjects
    all_betas = np.array([r['params'][1:] for r in results_full.values()])  # Shape: (n_subjects, 8)
    mean_betas = all_betas.mean(axis=0)  # Average weight at each frame
    sem_betas = all_betas.std(axis=0) / np.sqrt(len(all_betas))  # SEM
    
    # Bar positions
    x_pos = np.arange(1, 9)
    
    # Determine bar width based on number of models
    n_models = 1  # Start with full model
    if results_leaky is not None:
        n_models += 1
    if results_si_leaky is not None:
        n_models += 1
    
    bar_width = 0.7 / n_models if n_models > 1 else 0.7
    offset = -(n_models - 1) * bar_width / 2  # Starting offset
    
    # Bar plot of full model weights
    ax2.bar(
        x_pos + offset,
        mean_betas,
        width=bar_width,
        yerr=sem_betas,
        color='teal', 
        edgecolor='black', 
        capsize=3,
        label='Full model (free weights)'
    )
    offset += bar_width
    
    # Add leaky model implied weights (if provided)
    if results_leaky is not None:
        all_implied_leaky = []
        
        for r in results_leaky.values():
            b_subj = r['params'][1]    # Slope
            lam_subj = r['params'][2]  # Leak parameter
            weights_subj = b_subj * np.array([lam_subj ** (7 - i) for i in range(8)])
            all_implied_leaky.append(weights_subj)
        
        all_implied_leaky = np.array(all_implied_leaky)
        mean_implied_leaky = all_implied_leaky.mean(axis=0)
        sem_implied_leaky = all_implied_leaky.std(axis=0) / np.sqrt(len(all_implied_leaky))
        mean_lambda = np.array([r['params'][2] for r in results_leaky.values()]).mean()
        
        ax2.bar(
            x_pos + offset,
            mean_implied_leaky,
            width=bar_width,
            yerr=sem_implied_leaky,
            color='orange',
            edgecolor='black',
            capsize=3,
            label=f'Leaky model (λ={mean_lambda:.3f})'
        )
        offset += bar_width
    
    # Add SI-Leaky model implied weights (if provided)
    if results_si_leaky is not None:
        # For SI-Leaky, compute effective weights combining SI and temporal effects
        all_implied_si_leaky = []
        
        seqA_cols = [f'sequenceA{i}' for i in range(1, 9)]
        seqB_cols = [f'sequenceB{i}' for i in range(1, 9)]
        
        for subj_id, r in results_si_leaky.items():
            b_subj = r['params'][1]     # Slope
            beta_subj = r['params'][2]  # SI selectivity
            lam_subj = r['params'][3]   # Leak parameter
            
            # Get this subject's data from ALL trials
            df_subj = df_all[df_all['subject'] == subj_id]
            seqA_matrix = df_subj[seqA_cols].values
            seqB_matrix = df_subj[seqB_cols].values
            
            # Compute average SI amplification per frame
            diff_matrix = seqA_matrix - seqB_matrix
            abs_diff = np.abs(diff_matrix)
            
            # Average SI amplification: how much does SI boost the signal?
            # w_higher - w_lower tells us the net amplification
            w_higher = expit(beta_subj * abs_diff)
            w_lower = 1 - w_higher
            # Net amplification per frame = 2 * (w_higher - 0.5) = 2 * w_higher - 1
            # This ranges from -1 (when w_higher=0) to +1 (when w_higher=1)
            # At β=0: w_higher=0.5, so net_amplification=0 (no effect)
            net_si_amplification = 2 * w_higher - 1
            mean_net_si = net_si_amplification.mean(axis=0)  # Average across trials per frame
            
            # Temporal weights
            temporal_weights = np.array([lam_subj ** (7 - i) for i in range(8)])
            
            # Effective weight: baseline temporal weight boosted by SI
            # Using (1 + mean_net_si) to keep it positive and centered at 1
            effective_weights = b_subj * temporal_weights * (1 + mean_net_si)
            all_implied_si_leaky.append(effective_weights)
        
        all_implied_si_leaky = np.array(all_implied_si_leaky)
        mean_implied_si_leaky = all_implied_si_leaky.mean(axis=0)
        sem_implied_si_leaky = all_implied_si_leaky.std(axis=0) / np.sqrt(len(all_implied_si_leaky))
        
        mean_beta = np.array([r['params'][2] for r in results_si_leaky.values()]).mean()
        mean_lambda_si = np.array([r['params'][3] for r in results_si_leaky.values()]).mean()
        
        ax2.bar(
            x_pos + offset,
            mean_implied_si_leaky,
            width=bar_width,
            yerr=sem_implied_si_leaky,
            color='purple',
            edgecolor='black',
            capsize=3,
            label=f'SI-Leaky (β={mean_beta:.3f}, λ={mean_lambda_si:.3f})'
        )
    
    # Reference lines
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)  # Zero = no effect
    ax2.axhline(np.mean(mean_betas), color='red', linestyle='--', 
                label=f'Mean (full) = {np.mean(mean_betas):.4f}')
    
    ax2.set_xlabel('Frame Position')
    ax2.set_ylabel('Beta Weight (± SEM)')
    ax2.set_title('Frame-by-Frame Weights')
    ax2.set_xticks(range(1, 9))
    ax2.legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()


def print_results(accuracy, pref_broad, results_simple, results_full, comparison, 
                  results_leaky=None, results_si_leaky=None):
    """
    Print formatted summary of all analysis results.
    """
    print("=" * 60)
    print("DESCRIPTIVE RESULTS")
    print("=" * 60)
    print(f"Accuracy (different-mean trials): {accuracy:.1f}%")
    print(f"Preference for broader (same-mean trials): {pref_broad:.1f}%")
    print(f"Number of subjects: {comparison['n_subjects']}")
    print(f"Total trials: {comparison['n_obs']}")
    
    # Compute mean and SD of parameters across subjects
    all_params_simple = np.array([r['params'] for r in results_simple.values()])
    all_params_full = np.array([r['params'] for r in results_full.values()])
    
    mean_simple = all_params_simple.mean(axis=0)
    std_simple = all_params_simple.std(axis=0)
    mean_full = all_params_full.mean(axis=0)
    std_full = all_params_full.std(axis=0)
    
    # Simple model results
    print("\n" + "=" * 60)
    print("MODEL 1: P(choose A) = logistic(a + b * meanDiff)")
    print("=" * 60)
    print(f"a (intercept) = {mean_simple[0]:.4f} ± {std_simple[0]:.4f}")
    print(f"b (slope)     = {mean_simple[1]:.4f} ± {std_simple[1]:.4f}")
    print(f"Total log-likelihood = {comparison['ll_simple']:.2f}")
    
    # Full model results
    print("\n" + "=" * 60)
    print("MODEL 2: P(choose A) = logistic(a + b1*diff1 + ... + b8*diff8)")
    print("=" * 60)
    print(f"a (intercept) = {mean_full[0]:.4f} ± {std_full[0]:.4f}")
    for i in range(8):
        print(f"b{i+1} (frame {i+1}) = {mean_full[i+1]:.4f} ± {std_full[i+1]:.4f}")
    print(f"Total log-likelihood = {comparison['ll_full']:.2f}")
    
    # Leaky model results (if provided)
    if results_leaky is not None:
        all_params_leaky = np.array([r['params'] for r in results_leaky.values()])
        mean_leaky = all_params_leaky.mean(axis=0)
        std_leaky = all_params_leaky.std(axis=0)
        
        print("\n" + "=" * 60)
        print("MODEL 3: P(choose A) = logistic(a + b * Σ λ^(8-i) * diff_i)")
        print("=" * 60)
        print(f"a (intercept) = {mean_leaky[0]:.4f} ± {std_leaky[0]:.4f}")
        print(f"b (slope)     = {mean_leaky[1]:.4f} ± {std_leaky[1]:.4f}")
        print(f"λ (leak)      = {mean_leaky[2]:.4f} ± {std_leaky[2]:.4f}")
        print(f"Total log-likelihood = {comparison['ll_leaky']:.2f}")
        
        if mean_leaky[2] < 0.95:
            print(f"\nInterpretation: λ < 1 indicates recency effect (later frames weighted more)")
        else:
            print(f"\nInterpretation: λ ≈ 1 indicates approximately equal weighting")
    
    # SI-Leaky model results (if provided)
    if results_si_leaky is not None:
        all_params_si_leaky = np.array([r['params'] for r in results_si_leaky.values()])
        mean_si_leaky = all_params_si_leaky.mean(axis=0)
        std_si_leaky = all_params_si_leaky.std(axis=0)
        
        print("\n" + "=" * 60)
        print("MODEL 4: Hybrid SI-Leaky")
        print("P(choose A) = logistic(a + b * Σ λ^(8-i) * sigmoid(β*|diff_i|) * diff_i)")
        print("=" * 60)
        print(f"a (intercept) = {mean_si_leaky[0]:.4f} ± {std_si_leaky[0]:.4f}")
        print(f"b (slope)     = {mean_si_leaky[1]:.4f} ± {std_si_leaky[1]:.4f}")
        print(f"β (SI select) = {mean_si_leaky[2]:.4f} ± {std_si_leaky[2]:.4f}")
        print(f"λ (leak)      = {mean_si_leaky[3]:.4f} ± {std_si_leaky[3]:.4f}")
        print(f"Total log-likelihood = {comparison['ll_si_leaky']:.2f}")
        
        print(f"\nInterpretation:")
        if mean_si_leaky[2] > 0.05:
            print(f"  β > 0: Value-based selective integration present")
        else:
            print(f"  β ≈ 0: Minimal value-based selection")
        if mean_si_leaky[3] < 0.95:
            print(f"  λ < 1: Recency effect (temporal decay)")
        else:
            print(f"  λ ≈ 1: Equal temporal weighting")
    
    # Model comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"Simple   (k=2): AIC = {comparison['aic_simple']:.2f}, BIC = {comparison['bic_simple']:.2f}")
    print(f"Full     (k=9): AIC = {comparison['aic_full']:.2f}, BIC = {comparison['bic_full']:.2f}")
    
    aic_vals = {'Simple': comparison['aic_simple'], 'Full': comparison['aic_full']}
    bic_vals = {'Simple': comparison['bic_simple'], 'Full': comparison['bic_full']}
    
    if results_leaky is not None:
        print(f"Leaky    (k=3): AIC = {comparison['aic_leaky']:.2f}, BIC = {comparison['bic_leaky']:.2f}")
        aic_vals['Leaky'] = comparison['aic_leaky']
        bic_vals['Leaky'] = comparison['bic_leaky']
    
    if results_si_leaky is not None:
        print(f"SI-Leaky (k=4): AIC = {comparison['aic_si_leaky']:.2f}, BIC = {comparison['bic_si_leaky']:.2f}")
        aic_vals['SI-Leaky'] = comparison['aic_si_leaky']
        bic_vals['SI-Leaky'] = comparison['bic_si_leaky']
    
    aic_winner = min(aic_vals, key=aic_vals.get)
    bic_winner = min(bic_vals, key=bic_vals.get)
    print(f"\nPreferred by AIC: {aic_winner} model")
    print(f"Preferred by BIC: {bic_winner} model")

def neg_log_likelihood_si_leaky_theta(params, seqA_matrix, seqB_matrix, y):
    """
    Negative log-likelihood for modified SI-Leaky model with theta.
    
    This variant gives LOWER values more weight (opposite of standard SI).
    The theta parameter controls the strength of this "attention to worse" effect.
    
    SI weighting operates on individual values within each frame:
    - LOWER value gets weight w = sigmoid(β * |A_i - B_i|) * theta
    - HIGHER value gets weight (1 - w)
    - theta ∈ [0,1] scales the attention to lower values
    
    Model: P(choose A) = logistic(a + b * weighted_sum)
    where weighted_sum combines SI (lower-value attention) and temporal weighting.
    
    Args:
        params: [a, b, β, λ, θ] - intercept, slope, SI selectivity, leak parameter, theta
        seqA_matrix: (n_trials, 8) array of sequence A values
        seqB_matrix: (n_trials, 8) array of sequence B values
        y: Binary outcomes (1 = chose A, 0 = chose B)
        
    Returns:
        Negative log-likelihood (scalar to minimize)
    """
    # Unpack parameters
    a = params[0]      # Intercept (bias toward A or B)
    b = params[1]      # Slope (sensitivity to evidence)
    beta = params[2]   # SI selectivity (value-dependent attention)
    lam = params[3]    # Leak parameter (0 to 1) for temporal decay
    theta = params[4]  # Theta parameter (0 to 1) - scales attention to lower values
    
    # Compute differences to determine which value is lower in each frame
    diff_matrix = seqA_matrix - seqB_matrix  # (n_trials, 8)
    abs_diff = np.abs(diff_matrix)
    
    # SI weighting: LOWER value gets more weight (scaled by theta)
    # w_lower = sigmoid(β * |difference|) * theta
    w_lower_base = expit(beta * abs_diff)  # (n_trials, 8)
    w_lower = w_lower_base * theta
    w_higher = 1 - w_lower
    
    # Apply weights to actual values based on which is lower
    # When A < B: weighted_diff = w_lower * A - w_higher * B (negative, favors B)
    # When B < A: weighted_diff = w_higher * A - w_lower * B (positive, favors A)
    A_is_lower = (diff_matrix < 0).astype(float)  # 1 when A < B, 0 otherwise
    
    # Weight for A: w_lower when A < B, w_higher when A > B
    w_A = w_lower * A_is_lower + w_higher * (1 - A_is_lower)
    # Weight for B: w_higher when A < B, w_lower when A > B  
    w_B = w_higher * A_is_lower + w_lower * (1 - A_is_lower)
    
    # Compute weighted differences
    si_weighted_diff = w_A * seqA_matrix - w_B * seqB_matrix  # (n_trials, 8)
    
    # Apply temporal weights (recency effect)
    temporal_weights = np.array([lam ** (7 - i) for i in range(8)])  # [λ^7, ..., λ^0]
    
    # Combine SI and temporal weighting
    weighted_sum = si_weighted_diff @ temporal_weights  # (n_trials,)
    
    # Linear predictor
    linear = a + b * weighted_sum
    
    # Convert to probability using logistic function
    p = expit(linear)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    
    # Compute negative log-likelihood
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_logistic_si_leaky_theta(df_all):
    """
    Fit the modified SI-Leaky integration model with theta parameter.
    
    This variant tests whether people attend MORE to LOWER values (perhaps as
    a cautionary mechanism) rather than higher values.
    
    Model: P(choose A) = logistic(a + b * weighted_sum)
    where weighted_sum combines:
    - SI weighting: LOWER values in each frame get more weight (scaled by θ)
    - Temporal weighting: λ^(8-i) creates recency effect
    
    The model has 5 parameters:
    - a: intercept/bias
    - b: sensitivity to evidence
    - β: SI selectivity (strength of value-dependent attention)
    - λ: temporal decay (0 to 1)
    - θ: theta scaling (0 to 1) - strength of "attention to worse" effect
    
    Args:
        df_all: All trials DataFrame
        
    Returns:
        Dict mapping subject ID to {params: [a, b, β, λ, θ], ll: log_likelihood, n: n_trials}
    """
    # Column names for the 8 frames in each sequence
    seqA_cols = [f'sequenceA{i}' for i in range(1, 9)]
    seqB_cols = [f'sequenceB{i}' for i in range(1, 9)]
    results = {}
    
    # Fit model separately for each subject
    for subj in df_all['subject'].unique():
        # Get this subject's data
        df_subj = df_all[df_all['subject'] == subj]
        
        # Extract actual sequence values as matrices
        seqA_matrix = df_subj[seqA_cols].values  # (n_trials, 8)
        seqB_matrix = df_subj[seqB_cols].values  # (n_trials, 8)
        
        # Outcome: did they choose A?
        y = df_subj['choseA'].values
        
        # Initial parameter guesses
        # a = 0: no initial bias
        # b = 0.1: moderate sensitivity
        # β = 0.1: small SI effect
        # λ = 0.9: slight recency effect
        # θ = 0.5: moderate theta (middle of range)
        init_params = [0, 0.1, 0.1, 0.9, 0.5]
        
        # Parameter bounds
        # a, b: unbounded
        # β: positive (SI amplification)
        # λ: (0, 1] for temporal decay
        # θ: [0, 1] scaling factor for lower-value attention
        bounds = [(None, None), (None, None), (1e-6, 10.0), (1e-6, 1.0), (1e-6, 1.0)]
        
        # Optimize
        result = minimize(
            neg_log_likelihood_si_leaky_theta,
            init_params,
            args=(seqA_matrix, seqB_matrix, y),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        # Store results
        results[subj] = {
            'params': result.x,   # [a, b, β, λ, θ]
            'll': -result.fun,
            'n': len(df_subj)
        }
    
    return results

# def main(data_path='dataFit.csv'):
#     """
#     Main analysis pipeline.
    
#     Executes the full analysis workflow:
#     1. Load and preprocess data
#     2. Split by trial type
#     3. Compute descriptive statistics
#     4. Fit logistic regression models (per subject)
#     5. Compare models
#     6. Print results and generate plots
    
#     Args:
#         data_path: Path to the CSV data file
        
#     Returns:
#         Dict containing all analysis results
#     """
#     # Step 1: Load data and compute derived variables
#     df = load_and_prepare_data(data_path)
    
#     # Step 2: Split into different trial types
#     df_diffMean, df_sameMean = split_by_condition(df)
    
#     # Step 3: Compute basic descriptive statistics
#     accuracy, pref_broad = compute_descriptives(df_diffMean, df_sameMean)
    
#     # Step 4: Fit logistic models to each subject on ALL trials
#     # Combine both trial types for comprehensive model fitting
#     df_all = pd.concat([df_diffMean, df_sameMean], ignore_index=True)
    
#     print("Fitting models on ALL trials (different-mean + same-mean)...")
#     print(f"  Total trials: {len(df_all)} ({len(df_diffMean)} different-mean, {len(df_sameMean)} same-mean)")
    
#     results_simple = fit_logistic_simple(df_all)      # Simple: mean-difference model
#     print("  Simple model fitted")
#     results_full = fit_logistic_full(df_all)          # Full: frame-by-frame model
#     print("  Full model fitted")
#     results_leaky = fit_logistic_leaky(df_all)        # Leaky: exponential decay model
#     print("  Leaky model fitted")
#     results_si_leaky = fit_logistic_si_leaky(df_all)  # SI-Leaky: hybrid model
#     print("  SI-Leaky model fitted")
    
#     # Step 5: Compare models using information criteria
#     comparison = compare_models(results_simple, results_full, results_leaky, results_si_leaky)
    
#     # Step 6: Output results
#     print_results(accuracy, pref_broad, results_simple, results_full, comparison, 
#                   results_leaky, results_si_leaky)
    
#     # Step 7: Generate visualizations (display only, no saving)
#     plot_descriptives(df_diffMean, df_sameMean)
#     plot_individual_accuracy(df_diffMean, df_sameMean)
#     plot_logistic_fits(df_all, results_simple, results_full, results_leaky, results_si_leaky)
#     plot_lambda_distribution(results_leaky, results_si_leaky)
#     plot_model_comparison(comparison)
    
#     # Return all results for further analysis if needed
#     return {
#         'accuracy': accuracy,
#         'pref_broad': pref_broad,
#         'results_simple': results_simple,
#         'results_full': results_full,
#         'results_leaky': results_leaky,
#         'results_si_leaky': results_si_leaky,
#         'comparison': comparison,
#         'df_diffMean': df_diffMean,
#         'df_sameMean': df_sameMean,
#         'df_all': df_all
#     }

def compare_models_with_theta(results_simple, results_full, results_leaky=None, 
                               results_si_leaky=None, results_si_leaky_theta=None):
    """
    Compare models using information criteria, including the theta model.
    
    AIC (Akaike): -2*LL + 2*k  (penalizes complexity less)
    BIC (Bayesian): -2*LL + k*log(n)  (penalizes complexity more)
    
    Lower values = better model (balancing fit and parsimony)
    
    Args:
        results_simple: Output from fit_logistic_simple
        results_full: Output from fit_logistic_full
        results_leaky: Output from fit_logistic_leaky (optional)
        results_si_leaky: Output from fit_logistic_si_leaky (optional)
        results_si_leaky_theta: Output from fit_logistic_si_leaky_theta (optional)
        
    Returns:
        Dict with AIC, BIC, and log-likelihood for all models
    """
    # Number of free parameters in each model
    k_simple = 2         # intercept + 1 slope
    k_full = 9           # intercept + 8 frame weights
    k_leaky = 3          # intercept + slope + lambda
    k_si_leaky = 4       # intercept + slope + beta + lambda
    k_si_leaky_theta = 5 # intercept + slope + beta + lambda + theta
    
    # Initialize accumulators
    aic_simple = 0
    aic_full = 0
    aic_leaky = 0
    aic_si_leaky = 0
    aic_si_leaky_theta = 0
    bic_simple = 0
    bic_full = 0
    bic_leaky = 0
    bic_si_leaky = 0
    bic_si_leaky_theta = 0
    ll_simple = 0
    ll_full = 0
    ll_leaky = 0
    ll_si_leaky = 0
    ll_si_leaky_theta = 0
    
    # Compute fit statistics per subject, then sum
    for subj in results_simple.keys():
        n_trials = results_simple[subj]['n']
        ll_s = results_simple[subj]['ll']
        ll_f = results_full[subj]['ll']
        
        ll_simple += ll_s
        ll_full += ll_f
        
        aic_simple += -2 * ll_s + 2 * k_simple
        aic_full += -2 * ll_f + 2 * k_full
        
        bic_simple += -2 * ll_s + k_simple * np.log(n_trials)
        bic_full += -2 * ll_f + k_full * np.log(n_trials)
        
        if results_leaky is not None:
            ll_l = results_leaky[subj]['ll']
            ll_leaky += ll_l
            aic_leaky += -2 * ll_l + 2 * k_leaky
            bic_leaky += -2 * ll_l + k_leaky * np.log(n_trials)
        
        if results_si_leaky is not None:
            ll_sl = results_si_leaky[subj]['ll']
            ll_si_leaky += ll_sl
            aic_si_leaky += -2 * ll_sl + 2 * k_si_leaky
            bic_si_leaky += -2 * ll_sl + k_si_leaky * np.log(n_trials)
        
        if results_si_leaky_theta is not None:
            ll_slt = results_si_leaky_theta[subj]['ll']
            ll_si_leaky_theta += ll_slt
            aic_si_leaky_theta += -2 * ll_slt + 2 * k_si_leaky_theta
            bic_si_leaky_theta += -2 * ll_slt + k_si_leaky_theta * np.log(n_trials)
    
    n_obs = sum(r['n'] for r in results_simple.values())
    n_subjects = len(results_simple)
    
    result = {
        'aic_simple': aic_simple, 'aic_full': aic_full,
        'bic_simple': bic_simple, 'bic_full': bic_full,
        'll_simple': ll_simple, 'll_full': ll_full,
        'n_subjects': n_subjects, 'n_obs': n_obs
    }
    
    if results_leaky is not None:
        result.update({
            'aic_leaky': aic_leaky,
            'bic_leaky': bic_leaky,
            'll_leaky': ll_leaky
        })
    
    if results_si_leaky is not None:
        result.update({
            'aic_si_leaky': aic_si_leaky,
            'bic_si_leaky': bic_si_leaky,
            'll_si_leaky': ll_si_leaky
        })
    
    if results_si_leaky_theta is not None:
        result.update({
            'aic_si_leaky_theta': aic_si_leaky_theta,
            'bic_si_leaky_theta': bic_si_leaky_theta,
            'll_si_leaky_theta': ll_si_leaky_theta
        })
    
    return result


def simulate_choices_leaky(params, diff_matrix):
    """
    Simulate choices using the leaky integration model.
    
    Args:
        params: [a, b, λ] from fitted model
        diff_matrix: (n_trials, 8) frame differences
        
    Returns:
        Array of choice probabilities for choosing A
    """
    a, b, lam = params
    
    # Temporal weights
    weights = np.array([lam ** (7 - i) for i in range(8)])
    
    # Weighted sum of evidence
    weighted_sum = diff_matrix @ weights
    
    # Choice probability
    p_choose_A = expit(a + b * weighted_sum)
    
    return p_choose_A


def simulate_choices_si_leaky(params, seqA_matrix, seqB_matrix):
    """
    Simulate choices using the SI-Leaky model.
    
    Args:
        params: [a, b, β, λ] from fitted model
        seqA_matrix: (n_trials, 8) sequence A values
        seqB_matrix: (n_trials, 8) sequence B values
        
    Returns:
        Array of choice probabilities for choosing A
    """
    a, b, beta, lam = params
    
    # Compute differences
    diff_matrix = seqA_matrix - seqB_matrix
    abs_diff = np.abs(diff_matrix)
    
    # SI weighting: higher value gets more weight
    w_higher = expit(beta * abs_diff)
    w_lower = 1 - w_higher
    
    # Apply weights based on which is higher
    A_is_higher = (diff_matrix > 0).astype(float)
    w_A = w_higher * A_is_higher + w_lower * (1 - A_is_higher)
    w_B = w_lower * A_is_higher + w_higher * (1 - A_is_higher)
    
    # SI weighted differences
    si_weighted_diff = w_A * seqA_matrix - w_B * seqB_matrix
    
    # Temporal weights
    temporal_weights = np.array([lam ** (7 - i) for i in range(8)])
    
    # Combine
    weighted_sum = si_weighted_diff @ temporal_weights
    
    # Choice probability
    p_choose_A = expit(a + b * weighted_sum)
    
    return p_choose_A


def simulate_choices_si_leaky_theta(params, seqA_matrix, seqB_matrix):
    """
    Simulate choices using the SI-Leaky-Theta model.
    
    Args:
        params: [a, b, β, λ, θ] from fitted model
        seqA_matrix: (n_trials, 8) sequence A values
        seqB_matrix: (n_trials, 8) sequence B values
        
    Returns:
        Array of choice probabilities for choosing A
    """
    a, b, beta, lam, theta = params
    
    # Compute differences
    diff_matrix = seqA_matrix - seqB_matrix
    abs_diff = np.abs(diff_matrix)
    
    # SI weighting: LOWER value gets more weight (scaled by theta)
    w_lower_base = expit(beta * abs_diff)
    w_lower = w_lower_base * theta
    w_higher = 1 - w_lower
    
    # Apply weights based on which is lower
    A_is_lower = (diff_matrix < 0).astype(float)
    w_A = w_lower * A_is_lower + w_higher * (1 - A_is_lower)
    w_B = w_higher * A_is_lower + w_lower * (1 - A_is_lower)
    
    # SI weighted differences
    si_weighted_diff = w_A * seqA_matrix - w_B * seqB_matrix
    
    # Temporal weights
    temporal_weights = np.array([lam ** (7 - i) for i in range(8)])
    
    # Combine
    weighted_sum = si_weighted_diff @ temporal_weights
    
    # Choice probability
    p_choose_A = expit(a + b * weighted_sum)
    
    return p_choose_A


def analyze_risk_preference_predictions(df_sameMean, results_leaky, results_si_leaky, 
                                        results_si_leaky_theta):
    """
    Simulate model predictions on same-mean trials and compare to human data.
    
    This analysis tests whether models correctly predict risk preferences.
    - Leaky model (no SI): should predict ~50% (no preference)
    - SI-Leaky: should predict human-like preference if model is correct
    - SI-Leaky-Theta: tests alternative attention mechanism
    
    Args:
        df_sameMean: Same-mean trials DataFrame
        results_leaky: Fitted leaky model parameters
        results_si_leaky: Fitted SI-leaky model parameters
        results_si_leaky_theta: Fitted SI-leaky-theta model parameters
        
    Returns:
        Dict with human and model predictions (mean and 95% CI)
    """
    # Column names
    diff_cols = [f'diff{i}' for i in range(1, 9)]
    seqA_cols = [f'sequenceA{i}' for i in range(1, 9)]
    seqB_cols = [f'sequenceB{i}' for i in range(1, 9)]
    
    # Storage for per-subject predictions
    human_prefs = []
    leaky_prefs = []
    si_leaky_prefs = []
    si_leaky_theta_prefs = []
    
    # Analyze each subject
    for subj in df_sameMean['subject'].unique():
        # Get this subject's same-mean trials
        df_subj = df_sameMean[df_sameMean['subject'] == subj]
        
        # Human preference for broader
        human_pref = df_subj['choseBroader'].mean() * 100
        human_prefs.append(human_pref)
        
        # Extract data matrices
        diff_matrix = df_subj[diff_cols].values
        seqA_matrix = df_subj[seqA_cols].values
        seqB_matrix = df_subj[seqB_cols].values
        A_is_broader = df_subj['AisBroader'].values
        
        # === Leaky Model Predictions ===
        params_leaky = results_leaky[subj]['params']
        p_choose_A_leaky = simulate_choices_leaky(params_leaky, diff_matrix)
        
        # Convert to probability of choosing broader
        # If A is broader: p_broader = p_choose_A
        # If B is broader: p_broader = 1 - p_choose_A
        p_choose_broader_leaky = np.where(A_is_broader, p_choose_A_leaky, 1 - p_choose_A_leaky)
        leaky_pref = p_choose_broader_leaky.mean() * 100
        leaky_prefs.append(leaky_pref)
        
        # === SI-Leaky Model Predictions ===
        params_si_leaky = results_si_leaky[subj]['params']
        p_choose_A_si = simulate_choices_si_leaky(params_si_leaky, seqA_matrix, seqB_matrix)
        p_choose_broader_si = np.where(A_is_broader, p_choose_A_si, 1 - p_choose_A_si)
        si_leaky_pref = p_choose_broader_si.mean() * 100
        si_leaky_prefs.append(si_leaky_pref)
        
        # === SI-Leaky-Theta Model Predictions ===
        params_theta = results_si_leaky_theta[subj]['params']
        p_choose_A_theta = simulate_choices_si_leaky_theta(params_theta, seqA_matrix, seqB_matrix)
        p_choose_broader_theta = np.where(A_is_broader, p_choose_A_theta, 1 - p_choose_A_theta)
        si_leaky_theta_pref = p_choose_broader_theta.mean() * 100
        si_leaky_theta_prefs.append(si_leaky_theta_pref)
    
    # Convert to arrays
    human_prefs = np.array(human_prefs)
    leaky_prefs = np.array(leaky_prefs)
    si_leaky_prefs = np.array(si_leaky_prefs)
    si_leaky_theta_prefs = np.array(si_leaky_theta_prefs)
    
    # Compute means and 95% CIs
    def compute_ci(data):
        """Compute mean and 95% confidence interval."""
        mean = data.mean()
        sem = data.std() / np.sqrt(len(data))
        ci = 1.96 * sem  # 95% CI
        return mean, ci
    
    human_mean, human_ci = compute_ci(human_prefs)
    leaky_mean, leaky_ci = compute_ci(leaky_prefs)
    si_leaky_mean, si_leaky_ci = compute_ci(si_leaky_prefs)
    si_leaky_theta_mean, si_leaky_theta_ci = compute_ci(si_leaky_theta_prefs)
    
    results = {
        'human': {'mean': human_mean, 'ci': human_ci, 'data': human_prefs},
        'leaky': {'mean': leaky_mean, 'ci': leaky_ci, 'data': leaky_prefs},
        'si_leaky': {'mean': si_leaky_mean, 'ci': si_leaky_ci, 'data': si_leaky_prefs},
        'si_leaky_theta': {'mean': si_leaky_theta_mean, 'ci': si_leaky_theta_ci, 'data': si_leaky_theta_prefs}
    }
    
    return results


def plot_risk_preference_comparison(risk_pref_results):
    """
    Plot comparison of human vs model predictions for risk preference.
    
    Shows:
    - Human actual preference (with 95% CI)
    - Leaky model prediction (should be ~50%)
    - SI-Leaky model prediction
    - SI-Leaky-Theta model prediction
    
    Args:
        risk_pref_results: Output from analyze_risk_preference_predictions
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    models = ['Human\nActual', 'Leaky\n(no SI)', 'SI-Leaky\n(higher-value)', 'SI-Leaky-Theta\n(lower-value)']
    means = [
        risk_pref_results['human']['mean'],
        risk_pref_results['leaky']['mean'],
        risk_pref_results['si_leaky']['mean'],
        risk_pref_results['si_leaky_theta']['mean']
    ]
    cis = [
        risk_pref_results['human']['ci'],
        risk_pref_results['leaky']['ci'],
        risk_pref_results['si_leaky']['ci'],
        risk_pref_results['si_leaky_theta']['ci']
    ]
    colors = ['steelblue', 'orange', 'purple', 'crimson']
    
    # Create bar plot
    x_pos = np.arange(len(models))
    bars = ax.bar(x_pos, means, yerr=cis, color=colors, edgecolor='black', 
                  capsize=10, alpha=0.8, width=0.6)
    
    # Reference line at 50% (risk-neutral)
    ax.axhline(50, color='gray', linestyle='--', linewidth=2, label='Risk-neutral (50%)')
    
    # Labels and formatting
    ax.set_ylabel('Preference for Broader/Riskier Option (%)', fontsize=12, fontweight='bold')
    ax.set_title('Risk Preference: Human Data vs Model Predictions', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    
    # Add value labels on bars
    for i, (bar, mean, ci) in enumerate(zip(bars, means, cis)):
        ax.text(bar.get_x() + bar.get_width()/2, mean + ci + 2,
                f'{mean:.1f}%\n±{ci:.1f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    # Add grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("RISK PREFERENCE ANALYSIS: Human vs Model Predictions")
    print("=" * 70)
    print(f"Human (actual):          {means[0]:.1f}% ± {cis[0]:.1f}%")
    print(f"Leaky (no SI):           {means[1]:.1f}% ± {cis[1]:.1f}%  [Expected: ~50%]")
    print(f"SI-Leaky (higher-value): {means[2]:.1f}% ± {cis[2]:.1f}%")
    print(f"SI-Leaky-Theta (lower):  {means[3]:.1f}% ± {cis[3]:.1f}%")
    print("=" * 70)
    
    # Interpretation
    print("\nInterpretation:")
    print(f"- Leaky model predicts ~{means[1]:.1f}%, close to chance (no risk preference)")
    print(f"- SI-Leaky predicts {means[2]:.1f}%, {'matches' if abs(means[2] - means[0]) < 5 else 'differs from'} human {means[0]:.1f}%")
    print(f"- SI-Leaky-Theta predicts {means[3]:.1f}%, {'matches' if abs(means[3] - means[0]) < 5 else 'differs from'} human {means[0]:.1f}%")
    
    # Statistical comparison (simple difference test)
    from scipy import stats
    
    print("\n" + "=" * 70)
    print("Statistical Comparisons (paired t-tests vs human data):")
    print("=" * 70)
    
    human_data = risk_pref_results['human']['data']
    
    # Leaky vs Human
    t_leaky, p_leaky = stats.ttest_rel(human_data, risk_pref_results['leaky']['data'])
    print(f"Leaky vs Human:       t = {t_leaky:.3f}, p = {p_leaky:.4f}")
    
    # SI-Leaky vs Human
    t_si, p_si = stats.ttest_rel(human_data, risk_pref_results['si_leaky']['data'])
    print(f"SI-Leaky vs Human:    t = {t_si:.3f}, p = {p_si:.4f}")
    
    # SI-Leaky-Theta vs Human
    t_theta, p_theta = stats.ttest_rel(human_data, risk_pref_results['si_leaky_theta']['data'])
    print(f"SI-Leaky-Theta vs Human: t = {t_theta:.3f}, p = {p_theta:.4f}")
    
    print("=" * 70)

# def main(data_path='dataFit.csv'):
#     """
#     Main analysis pipeline.
    
#     Executes the full analysis workflow:
#     1. Load and preprocess data
#     2. Split by trial type
#     3. Compute descriptive statistics
#     4. Fit logistic regression models (per subject)
#     5. Compare models
#     6. Print results and generate plots
    
#     Args:
#         data_path: Path to the CSV data file
        
#     Returns:
#         Dict containing all analysis results
#     """
#     # Step 1: Load data and compute derived variables
#     df = load_and_prepare_data(data_path)
    
#     # Step 2: Split into different trial types
#     df_diffMean, df_sameMean = split_by_condition(df)
    
#     # Step 3: Compute basic descriptive statistics
#     accuracy, pref_broad = compute_descriptives(df_diffMean, df_sameMean)
    
#     # Step 4: Fit logistic models to each subject on ALL trials
#     # Combine both trial types for comprehensive model fitting
#     df_all = pd.concat([df_diffMean, df_sameMean], ignore_index=True)
    
#     print("Fitting models on ALL trials (different-mean + same-mean)...")
#     print(f"  Total trials: {len(df_all)} ({len(df_diffMean)} different-mean, {len(df_sameMean)} same-mean)")
    
#     results_simple = fit_logistic_simple(df_all)      # Simple: mean-difference model
#     print("  Simple model fitted")
#     results_full = fit_logistic_full(df_all)          # Full: frame-by-frame model
#     print("  Full model fitted")
#     results_leaky = fit_logistic_leaky(df_all)        # Leaky: exponential decay model
#     print("  Leaky model fitted")
#     results_si_leaky = fit_logistic_si_leaky(df_all)  # SI-Leaky: hybrid model
#     print("  SI-Leaky model fitted")
#     results_si_leaky_theta = fit_logistic_si_leaky_theta(df_all)  # SI-Leaky-Theta: modified SI
#     print("  SI-Leaky-Theta model fitted")
    
#     # Step 5: Compare models using information criteria
#     comparison = compare_models(results_simple, results_full, results_leaky, results_si_leaky)
    
#     # Step 6: Output results
#     print_results(accuracy, pref_broad, results_simple, results_full, comparison, 
#                   results_leaky, results_si_leaky)
    
#     # Print theta model results separately for now
#     all_params_theta = np.array([r['params'] for r in results_si_leaky_theta.values()])
#     mean_theta = all_params_theta.mean(axis=0)
#     std_theta = all_params_theta.std(axis=0)
    
#     print("\n" + "=" * 60)
#     print("MODEL 5: Modified SI-Leaky with Theta (Lower-Value Attention)")
#     print("P(choose A) = logistic(a + b * Σ λ^(8-i) * SI_theta(diff_i))")
#     print("=" * 60)
#     print(f"a (intercept) = {mean_theta[0]:.4f} ± {std_theta[0]:.4f}")
#     print(f"b (slope)     = {mean_theta[1]:.4f} ± {std_theta[1]:.4f}")
#     print(f"β (SI select) = {mean_theta[2]:.4f} ± {std_theta[2]:.4f}")
#     print(f"λ (leak)      = {mean_theta[3]:.4f} ± {std_theta[3]:.4f}")
#     print(f"θ (theta)     = {mean_theta[4]:.4f} ± {std_theta[4]:.4f}")
    
#     print(f"\nInterpretation:")
#     print(f"  θ = {mean_theta[4]:.3f}: Scaling of attention to LOWER values")
#     print(f"  θ → 0: No preference for lower values")
#     print(f"  θ → 1: Strong attention to lower/worse values")
    
#     # Step 7: Generate visualizations (display only, no saving)
#     plot_descriptives(df_diffMean, df_sameMean)
#     plot_individual_accuracy(df_diffMean, df_sameMean)
#     plot_logistic_fits(df_all, results_simple, results_full, results_leaky, results_si_leaky)
#     plot_lambda_distribution(results_leaky, results_si_leaky)
#     plot_model_comparison(comparison)
    
#     # Return all results for further analysis if needed
#     return {
#         'accuracy': accuracy,
#         'pref_broad': pref_broad,
#         'results_simple': results_simple,
#         'results_full': results_full,
#         'results_leaky': results_leaky,
#         'results_si_leaky': results_si_leaky,
#         'results_si_leaky_theta': results_si_leaky_theta,
#         'comparison': comparison,
#         'df_diffMean': df_diffMean,
#         'df_sameMean': df_sameMean,
#         'df_all': df_all
#     }

def main(data_path='dataFit.csv'):
    """
    Main analysis pipeline.
    """
    # Steps 1-4: Same as before
    df = load_and_prepare_data(data_path)
    df_diffMean, df_sameMean = split_by_condition(df)
    accuracy, pref_broad = compute_descriptives(df_diffMean, df_sameMean)
    df_all = pd.concat([df_diffMean, df_sameMean], ignore_index=True)
    
    print("Fitting models on ALL trials (different-mean + same-mean)...")
    print(f"  Total trials: {len(df_all)} ({len(df_diffMean)} different-mean, {len(df_sameMean)} same-mean)")
    
    results_simple = fit_logistic_simple(df_all)
    print("  Simple model fitted")
    results_full = fit_logistic_full(df_all)
    print("  Full model fitted")
    results_leaky = fit_logistic_leaky(df_all)
    print("  Leaky model fitted")
    results_si_leaky = fit_logistic_si_leaky(df_all)
    print("  SI-Leaky model fitted")
    results_si_leaky_theta = fit_logistic_si_leaky_theta(df_all)
    print("  SI-Leaky-Theta model fitted")
    
    # Step 5: Compare ALL models including theta
    comparison = compare_models_with_theta(results_simple, results_full, results_leaky, 
                                           results_si_leaky, results_si_leaky_theta)
    
    # Step 6: Print results
    print_results(accuracy, pref_broad, results_simple, results_full, comparison, 
                  results_leaky, results_si_leaky)
    
    # Print theta model results
    all_params_theta = np.array([r['params'] for r in results_si_leaky_theta.values()])
    mean_theta = all_params_theta.mean(axis=0)
    std_theta = all_params_theta.std(axis=0)
    
    print("\n" + "=" * 60)
    print("MODEL 5: Modified SI-Leaky with Theta (Lower-Value Attention)")
    print("=" * 60)
    print(f"a (intercept) = {mean_theta[0]:.4f} ± {std_theta[0]:.4f}")
    print(f"b (slope)     = {mean_theta[1]:.4f} ± {std_theta[1]:.4f}")
    print(f"β (SI select) = {mean_theta[2]:.4f} ± {std_theta[2]:.4f}")
    print(f"λ (leak)      = {mean_theta[3]:.4f} ± {std_theta[3]:.4f}")
    print(f"θ (theta)     = {mean_theta[4]:.4f} ± {std_theta[4]:.4f}")
    print(f"Total log-likelihood = {comparison['ll_si_leaky_theta']:.2f}")
    
    # Print model comparison with theta
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (including Theta model)")
    print("=" * 60)
    print(f"Simple        (k=2): AIC = {comparison['aic_simple']:.2f}, BIC = {comparison['bic_simple']:.2f}")
    print(f"Full          (k=9): AIC = {comparison['aic_full']:.2f}, BIC = {comparison['bic_full']:.2f}")
    print(f"Leaky         (k=3): AIC = {comparison['aic_leaky']:.2f}, BIC = {comparison['bic_leaky']:.2f}")
    print(f"SI-Leaky      (k=4): AIC = {comparison['aic_si_leaky']:.2f}, BIC = {comparison['bic_si_leaky']:.2f}")
    print(f"SI-Leaky-Theta(k=5): AIC = {comparison['aic_si_leaky_theta']:.2f}, BIC = {comparison['bic_si_leaky_theta']:.2f}")
    
    # Determine winners
    all_aic = {
        'Simple': comparison['aic_simple'],
        'Full': comparison['aic_full'],
        'Leaky': comparison['aic_leaky'],
        'SI-Leaky': comparison['aic_si_leaky'],
        'SI-Leaky-Theta': comparison['aic_si_leaky_theta']
    }
    all_bic = {
        'Simple': comparison['bic_simple'],
        'Full': comparison['bic_full'],
        'Leaky': comparison['bic_leaky'],
        'SI-Leaky': comparison['bic_si_leaky'],
        'SI-Leaky-Theta': comparison['bic_si_leaky_theta']
    }
    
    aic_winner = min(all_aic, key=all_aic.get)
    bic_winner = min(all_bic, key=all_bic.get)
    
    print(f"\nPreferred by AIC: {aic_winner}")
    print(f"Preferred by BIC: {bic_winner}")
    
    # Step 7: Simulate risk preference predictions
    print("\n" + "=" * 60)
    print("SIMULATING MODEL PREDICTIONS ON SAME-MEAN TRIALS...")
    print("=" * 60)
    risk_pref_results = analyze_risk_preference_predictions(
        df_sameMean, results_leaky, results_si_leaky, results_si_leaky_theta
    )
    
    # Step 8: Generate visualizations
    plot_descriptives(df_diffMean, df_sameMean)
    plot_individual_accuracy(df_diffMean, df_sameMean)
    plot_logistic_fits(df_all, results_simple, results_full, results_leaky, results_si_leaky)
    plot_lambda_distribution(results_leaky, results_si_leaky)
    plot_model_comparison(comparison)
    plot_risk_preference_comparison(risk_pref_results)  # NEW PLOT
    
    return {
        'accuracy': accuracy,
        'pref_broad': pref_broad,
        'results_simple': results_simple,
        'results_full': results_full,
        'results_leaky': results_leaky,
        'si_leaky': results_si_leaky,
        'results_si_leaky_theta': results_si_leaky_theta,
        'comparison': comparison,
        'risk_pref_results': risk_pref_results,
        'df_diffMean': df_diffMean,
        'df_sameMean': df_sameMean,
        'df_all': df_all
    }

if __name__ == '__main__':
    # Run analysis with default data path
    results = main(data_path='dataFit.csv')