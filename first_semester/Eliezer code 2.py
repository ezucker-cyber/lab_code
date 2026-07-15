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
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit  # Logistic (sigmoid) function: 1 / (1 + exp(-x))
import seaborn as sns
import matplotlib.patches as patches


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


def fit_logistic_simple(df_diffMean, choose_col='choseA'):
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
        y = df_subj[choose_col].values
        
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


def compare_models(results_simple, results_full, results_leaky=None):
    """
    Compare models using information criteria.
    
    AIC (Akaike): -2*LL + 2*k  (penalizes complexity less)
    BIC (Bayesian): -2*LL + k*log(n)  (penalizes complexity more)
    
    Lower values = better model (balancing fit and parsimony)
    
    Args:
        results_simple: Output from fit_logistic_simple
        results_full: Output from fit_logistic_full
        results_leaky: Output from fit_logistic_leaky (optional)
        
    Returns:
        Dict with AIC, BIC, and log-likelihood for all models
    """
    # Number of free parameters in each model
    k_simple = 2   # intercept + 1 slope
    k_full = 9     # intercept + 8 frame weights
    k_leaky = 3    # intercept + slope + lambda
    
    # Initialize accumulators
    aic_simple = 0
    aic_full = 0
    aic_leaky = 0
    bic_simple = 0
    bic_full = 0
    bic_leaky = 0
    ll_simple = 0
    ll_full = 0
    ll_leaky = 0
    
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
    - If simple wins: only overall mean difference matters, no temporal effects
    
    Args:
        comparison: Dict from compare_models() containing LL, AIC, BIC for all models
    """
    # Check if leaky model results are available
    has_leaky = 'll_leaky' in comparison
    
    # Prepare data for plotting
    if has_leaky:
        models = ['Simple\n(k=2)', 'Full\n(k=9)', 'Leaky\n(k=3)']
        ll_vals = [comparison['ll_simple'], comparison['ll_full'], comparison['ll_leaky']]
        aic_vals = [comparison['aic_simple'], comparison['aic_full'], comparison['aic_leaky']]
        bic_vals = [comparison['bic_simple'], comparison['bic_full'], comparison['bic_leaky']]
        colors = ['steelblue', 'teal', 'orange']
    else:
        models = ['Simple\n(k=2)', 'Full\n(k=9)']
        ll_vals = [comparison['ll_simple'], comparison['ll_full']]
        aic_vals = [comparison['aic_simple'], comparison['aic_full']]
        bic_vals = [comparison['bic_simple'], comparison['bic_full']]
        colors = ['steelblue', 'teal']
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
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
    model_names = ['Simple', 'Full', 'Leaky'] if has_leaky else ['Simple', 'Full']
    print(f"\nModel comparison summary:")
    print(f"  Best fit (LL):  {model_names[best_ll_idx]}")
    print(f"  Best AIC:       {model_names[best_aic_idx]}")
    print(f"  Best BIC:       {model_names[best_bic_idx]}")


def plot_lambda_distribution(results_leaky):
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
    lambdas = np.array([r['params'][2] for r in results_leaky.values()])
    
    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # === Left Panel: Histogram ===
    # Shows the frequency distribution of λ values
    ax1 = axes[0]
    ax1.hist(
        lambdas, 
        bins=15,                    # Number of bins for histogram
        color='orange', 
        edgecolor='black', 
        alpha=0.7                   # Slight transparency
    )
    
    # Add vertical line at mean λ
    ax1.axvline(
        lambdas.mean(), 
        color='red', 
        linestyle='-', 
        lw=2, 
        label=f'Mean = {lambdas.mean():.3f}'
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
    ax1.legend()
    
    # === Right Panel: Individual points (sorted) ===
    # Shows each subject's λ value, sorted from lowest to highest
    # Useful for identifying outliers and seeing the full range
    ax2 = axes[1]
    
    # Sort λ values for cleaner visualization
    lambdas_sorted = np.sort(lambdas)
    
    # Plot as connected dots (same style as accuracy plots)
    ax2.plot(
        range(len(lambdas_sorted)),  # X: subject rank (0 to N-1)
        lambdas_sorted,               # Y: sorted λ value
        'o-',                         # Circle markers connected by lines
        color='orange', 
        markersize=8, 
        markerfacecolor='orange', 
        markeredgecolor='black', 
        linewidth=1.5
    )
    
    # Reference lines (same as histogram)
    ax2.axhline(
        lambdas.mean(), 
        color='red', 
        linestyle='-', 
        lw=2, 
        label=f'Mean = {lambdas.mean():.3f}'
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
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics to console
    # Provides quick numerical summary alongside the visual
    print(f"\nLambda summary: M = {lambdas.mean():.3f}, SD = {lambdas.std():.3f}, "
          f"range = [{lambdas.min():.3f}, {lambdas.max():.3f}]")


def plot_logistic_fits(df_diffMean, results_simple, results_full, results_leaky=None):
    """
    Visualize the fitted logistic models.
    
    Left panel: Psychometric curves from simple model (choice probability vs mean difference)
    Right panel: Frame weights from full model (tests primacy/recency effects)
                 + implied weights from leaky model (if provided)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
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
    bar_width = 0.35
    
    # Bar plot of full model weights
    ax2.bar(
        x_pos - bar_width/2 if results_leaky else x_pos,  # Shift if leaky model present
        mean_betas,
        width=bar_width if results_leaky else 0.7,
        yerr=sem_betas,
        color='teal', 
        edgecolor='black', 
        capsize=3,
        label='Full model (free weights)'
    )
    
    # Add leaky model implied weights (if provided)
    if results_leaky is not None:
        # Compute implied weights for EACH subject individually, then average
        # This is the correct approach (vs computing from mean params)
        # because the relationship between b, λ, and weights is nonlinear
        all_implied_weights = []
        
        for r in results_leaky.values():
            # Extract this subject's fitted parameters
            b_subj = r['params'][1]    # Slope (sensitivity)
            lam_subj = r['params'][2]  # Leak parameter
            
            # Compute this subject's implied weight at each frame
            # Weight for frame i = b * λ^(8-i)
            # Frame 1: b * λ^7 (most discounted if λ < 1)
            # Frame 8: b * λ^0 = b (no discounting)
            weights_subj = b_subj * np.array([lam_subj ** (7 - i) for i in range(8)])
            all_implied_weights.append(weights_subj)
        
        # Stack into (n_subjects, 8) array for computing stats
        all_implied_weights = np.array(all_implied_weights)
        
        # Mean and SEM across subjects for each frame position
        mean_implied = all_implied_weights.mean(axis=0)
        sem_implied = all_implied_weights.std(axis=0) / np.sqrt(len(all_implied_weights))
        
        # Get mean lambda for the legend label
        mean_lambda = np.array([r['params'][2] for r in results_leaky.values()]).mean()
        
        # Plot leaky model implied weights as bars (next to full model bars)
        ax2.bar(
            x_pos + bar_width/2,     # Position to the right of full model bars
            mean_implied,
            width=bar_width,
            yerr=sem_implied,        # Error bars = SEM
            color='orange',
            edgecolor='black',
            capsize=3,
            label=f'Leaky model (λ={mean_lambda:.3f})'
        )
    
    # Reference lines
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)  # Zero = no effect
    ax2.axhline(np.mean(mean_betas), color='red', linestyle='--', 
                label=f'Mean (full) = {np.mean(mean_betas):.4f}')
    
    ax2.set_xlabel('Frame Position')
    ax2.set_ylabel('Beta Weight (± SEM)')
    ax2.set_title('Frame-by-Frame Weights')
    ax2.set_xticks(range(1, 9))
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def print_results(accuracy, pref_broad, results_simple, results_full, comparison, results_leaky=None):
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
        
        # Interpretation of lambda
        if mean_leaky[2] < 0.95:
            print(f"\nInterpretation: λ < 1 indicates recency effect (later frames weighted more)")
        else:
            print(f"\nInterpretation: λ ≈ 1 indicates approximately equal weighting")
    
    # Model comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"Simple (k=2): AIC = {comparison['aic_simple']:.2f}, BIC = {comparison['bic_simple']:.2f}")
    print(f"Full   (k=9): AIC = {comparison['aic_full']:.2f}, BIC = {comparison['bic_full']:.2f}")
    
    if results_leaky is not None:
        print(f"Leaky  (k=3): AIC = {comparison['aic_leaky']:.2f}, BIC = {comparison['bic_leaky']:.2f}")
        
        # Find best model
        aic_vals = {'Simple': comparison['aic_simple'], 'Full': comparison['aic_full'], 'Leaky': comparison['aic_leaky']}
        bic_vals = {'Simple': comparison['bic_simple'], 'Full': comparison['bic_full'], 'Leaky': comparison['bic_leaky']}
    else:
        aic_vals = {'Simple': comparison['aic_simple'], 'Full': comparison['aic_full']}
        bic_vals = {'Simple': comparison['bic_simple'], 'Full': comparison['bic_full']}
    
    aic_winner = min(aic_vals, key=aic_vals.get)
    bic_winner = min(bic_vals, key=bic_vals.get)
    print(f"\nPreferred by AIC: {aic_winner} model")
    print(f"Preferred by BIC: {bic_winner} model")


def add_params_and_prob(df, choose_col='choseA'):
    """
    Add columns 'a', 'b', and 'p_chooseA' to df based on subject-specific logistic parameters
    using the observed responses.

    Args:
        df: DataFrame with columns ['subject', 'meanDiff', choose_col]
        choose_col: column with observed choices (0/1)

    Returns:
        df with new columns 'a', 'b', 'p_chooseA'
    """
    results = fit_logistic_simple(df, choose_col=choose_col)
    df = df.copy()
    df['a'] = df['subject'].map(lambda s: results[s]['params'][0])
    df['b'] = df['subject'].map(lambda s: results[s]['params'][1])
    df['p_chooseA'] = 1 / (1 + np.exp(-(df['a'] + df['b'] * df['meanDiff'])))
    return df


def simulate_and_fit(df, n_trials=50):
    """
    Simulate Bernoulli responses multiple times, refit logistic model per subject,
    and store resulting 'a' and 'b' parameters for each trial.
    The first row for each subject is the original fitted parameters.

    Returns:
        results_dict: dict where keys = subjects, values = DataFrames with index=range(n_trials), columns=['a','b']
    """
    # Step 1: Fit initial model to get predicted probabilities and original params
    df_with_probs = add_params_and_prob(df)  # adds 'a', 'b', 'p_chooseA' per trial
    subjects = df['subject'].unique()

    # Initialize results dict
    results_dict = {subj: pd.DataFrame(columns=['a','b'], index=range(n_trials)) for subj in subjects}

    # Fill first row with original fitted params
    for subj in subjects:
        a_orig = df_with_probs.loc[df_with_probs['subject'] == subj, 'a'].iloc[0]
        b_orig = df_with_probs.loc[df_with_probs['subject'] == subj, 'b'].iloc[0]
        results_dict[subj].iloc[0] = [a_orig, b_orig]

    # Step 2: Loop over trials (starting from 1 since 0 is original)
    for trial in range(1, n_trials):
        # Simulate Bernoulli responses based on predicted probabilities
        df_sim = df_with_probs.copy()
        df_sim['sim_response'] = df_sim['p_chooseA'].apply(lambda p: np.random.binomial(1, p))
        
        # Fit logistic model for each subject
        results_trial = fit_logistic_simple(df_sim, choose_col='sim_response')
        
        for subj in subjects:
            a, b = results_trial[subj]['params']
            results_dict[subj].iloc[trial] = [a, b]
    
    return results_dict

def correlation_across_subj_and_trial(results_dict):
    """
    For each trial n, compute the correlation across subjects between
    original params (index 0) and recovered params (index n).

    Returns:
        corr_df: DataFrame with index=trial, columns=['corr_a', 'corr_b']
    """
    subjects = list(results_dict.keys())
    n_trials = len(next(iter(results_dict.values())))

    corr_df = pd.DataFrame(index=range(1, n_trials),
                           columns=['corr_a', 'corr_b'],
                           dtype=float)

    # Extract original parameters across subjects
    a_orig = np.array([results_dict[s].loc[0, 'a'] for s in subjects])
    b_orig = np.array([results_dict[s].loc[0, 'b'] for s in subjects])

    for trial in range(1, n_trials):
        a_rec = np.array([results_dict[s].loc[trial, 'a'] for s in subjects])
        b_rec = np.array([results_dict[s].loc[trial, 'b'] for s in subjects])

        corr_df.loc[trial, 'corr_a'] = np.corrcoef(a_orig, a_rec)[0, 1]
        corr_df.loc[trial, 'corr_b'] = np.corrcoef(b_orig, b_rec)[0, 1]

    return corr_df


def grid_search_logistic_mse(df, a_min=-1.0, a_max=1.0, b_min=0.0, b_max=0.5,  step=0.01):
    """
    Brute-force grid search over a and b for logistic model:
        p = sigmoid(a + b*x)
    
    Uses neg_log_likelihood between predicted probabilities and true y.

    Returns:
        DataFrame with columns [a, b, mse]
    """

    x = df['meanDiff'].values
    y = df['choseA'].values

    a_vals = np.arange(a_min, a_max + step, step)
    b_vals = np.arange(b_min, b_max + step, step)

    results = []

    for a in a_vals:
        for b in b_vals:
            z = a + (b * x)
            p = expit(z)
            neg_log_likelihood = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            results.append((a, b, neg_log_likelihood))

    results_df = pd.DataFrame(results, columns=['a', 'b', 'neg_log_likelihood'])
    return results_df

def grid_search_logistic_mse_by_subject(df, subject_col='subject', a_min=-2.0, a_max=2.0, b_min=0.0, b_max=1.0, step=0.01):
    """Run grid search separately for each subject, using grid_search_logistic_mse internally."""
    # TODO: scale all heatmaps by same values and set min max according to neg likelihood
    results_by_subject = {}
    
    for subject, df_sub in df.groupby(subject_col):
        results_by_subject[subject] = grid_search_logistic_mse(df_sub, a_min, a_max, b_min, b_max, step)
    
    return results_by_subject

def pretty_plot_grid_searches(
    results_by_subject,
    n_cols=11,
    cmap='YlGnBu',
    figsize_per_cell=2.4,
    title='Grid Search Likelihood Landscapes per Subject\n(Minimum NLL Boxed)'
):
    """
    Plot grid-search likelihood heatmaps for each subject in a tiled layout.
    Boxes the minimum NLL per subject and uses a shared color scale.
    """

    try:
        subjects = sorted(results_by_subject.keys())
    except TypeError:
        subjects = sorted(results_by_subject.keys(), key=str)

    n_subjects = len(subjects)
    n_rows = int(np.ceil(n_subjects / n_cols))

    all_nll = np.concatenate([
        res['neg_log_likelihood'].values
        for res in results_by_subject.values()
    ])
    vmin, vmax = all_nll.min(), all_nll.max()

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell * n_cols,
                 figsize_per_cell * n_rows),
        squeeze=False
    )

    for i, subject in enumerate(subjects):
        r = i // n_cols
        c = i % n_cols
        ax = axes[r, c]

        nll_df = results_by_subject[subject]

        heatmap = nll_df.pivot(
            index='a',
            columns='b',
            values='neg_log_likelihood'
        )

        sns.heatmap(
            heatmap,
            ax=ax,
            cmap=cmap,
            cbar=False,
            vmin=vmin,
            vmax=vmax
        )

        min_idx = np.unravel_index(
            np.argmin(heatmap.values),
            heatmap.values.shape
        )
        row, col = min_idx

        rect = patches.Rectangle(
            (col, row),
            1, 1,
            fill=False,
            edgecolor='red',
            linewidth=1.5
        )
        ax.add_patch(rect)

        ax.set_title(f'Subject {subject}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    for j in range(i + 1, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis('off')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='neg_log_likelihood')

    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()

def pretty_plot_grid_searches_ind(results_by_subject):
    """
    Plot individual heatmaps for each subject in results_by_subject.

    Each subject gets its own figure. The minimum negative log-likelihood
    is boxed in red.

    Parameters
    ----------
    results_by_subject : dict
        Keys are subject identifiers, values are DataFrames with columns
        ['a', 'b', 'neg_log_likelihood'] from grid search.
    """

    for subject, nll_df in results_by_subject.items():

        heatmap = nll_df.pivot(
            index='a',
            columns='b',
            values='neg_log_likelihood'
        )

        heatmap = heatmap.round(2)
        heatmap.index = heatmap.index.round(2)
        heatmap.columns = heatmap.columns.round(2)

        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        sns.heatmap(
            heatmap,
            annot=False,
            fmt=".2f",
            cmap="YlGnBu",
            cbar_kws={'label': 'neg_log_likelihood'},
            ax=ax
        )

        min_idx = np.unravel_index(
            np.argmin(heatmap.values),
            heatmap.values.shape
        )
        row, col = min_idx

        rect = patches.Rectangle(
            (col, row),
            1, 1,
            fill=False,
            edgecolor='red',
            linewidth=2
        )
        ax.add_patch(rect)

        plt.xlabel('b (slope)')
        plt.ylabel('a (intercept)')
        plt.title(f'Subject {subject}: neg_log_likelihood Heatmap')

        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.show()

def plot_hist(results_dict, index_of_dict=1):
    df_subj = results_dict[index_of_dict]

    # Plot histogram for 'a'
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(df_subj['a'][1:], bins=20, color='skyblue', edgecolor='black')  # exclude row 0
    plt.axvline(df_subj['a'].iloc[0], color='red', linestyle='--', label='Original a')
    plt.title("Histogram of 'a' (simulated)")
    plt.xlabel('a')
    plt.ylabel('Frequency')
    plt.legend()

    # Plot histogram for 'b'
    plt.subplot(1,2,2)
    plt.hist(df_subj['b'][1:], bins=20, color='lightgreen', edgecolor='black')  # exclude row 0
    plt.axvline(df_subj['b'].iloc[0], color='red', linestyle='--', label='Original b')
    plt.title("Histogram of 'b' (simulated)")
    plt.xlabel('b')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    df = load_and_prepare_data('dataFit.csv')
    df['subject'] = df['subject'].astype(int)

    df_diffMean, df_sameMean = split_by_condition(df)

    results_dict = simulate_and_fit(df_diffMean, n_trials=500)
    corr_df = correlation_across_subj_and_trial(results_dict)

    plt.figure()

    plt.hist(corr_df['corr_a'].dropna(), bins=40, alpha=0.6, label='corr_a')
    plt.hist(corr_df['corr_b'].dropna(), bins=40, alpha=0.6, label='corr_b')

    plt.xlabel('Correlation')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Parameter Recovery Correlations')

    plt.show()

    # for subj in results_dict.keys():
    #     plot_hist(results_dict, subj)
    # results_by_subject = grid_search_logistic_mse_by_subject(df_diffMean)

    # pretty_plot_grid_searches_ind(results_by_subject)

# Moshe's original main function
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
    
#     # Step 4: Fit logistic models to each subject
#     results_simple = fit_logistic_simple(df_diffMean)   # Simple: mean-difference model
#     results_full = fit_logistic_full(df_diffMean)       # Full: frame-by-frame model
#     results_leaky = fit_logistic_leaky(df_diffMean)     # Leaky: exponential decay model
    
#     # Step 5: Compare models using information criteria
#     comparison = compare_models(results_simple, results_full, results_leaky)
    
#     # Step 6: Output results
#     print_results(accuracy, pref_broad, results_simple, results_full, comparison, results_leaky)
    
#     # Step 7: Generate visualizations (display only, no saving)
#     plot_descriptives(df_diffMean, df_sameMean)
#     plot_individual_accuracy(df_diffMean, df_sameMean)
#     plot_logistic_fits(df_diffMean, results_simple, results_full, results_leaky)
#     plot_lambda_distribution(results_leaky)
#     plot_model_comparison(comparison)
    
#     # Return all results for further analysis if needed
#     return {
#         'accuracy': accuracy,
#         'pref_broad': pref_broad,
#         'results_simple': results_simple,
#         'results_full': results_full,
#         'results_leaky': results_leaky,
#         'comparison': comparison,
#         'df_diffMean': df_diffMean,
#         'df_sameMean': df_sameMean
#     }