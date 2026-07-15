import numpy as np

from parameters import MU_HIGH, MU_LOW, SIGMA_M, K

# ---------------------------------------------------------------------------
# Integrators
# ---------------------------------------------------------------------------

def perfect_integrator(d, epsilon, sigma):
    """
    d       : array of shape (n_frames,) — evidence each frame
    epsilon : array of shape (n_frames,) — shared internal noise
    sigma   : float — internal noise scale
    returns : final belief L(12)
    """
    L = 0.0
    for t in range(len(d)):
        L = L + d[t] + np.sqrt(2) * sigma * epsilon[t]
    return L


def leaky_integrator(d, epsilon, sigma, k):
    """
    d       : array of shape (n_frames,) — evidence each frame
    epsilon : array of shape (n_frames,) — shared internal noise
    sigma   : float — internal noise scale
    k       : float — leak rate
    returns : final belief L(12)
    """
    L = 0.0
    for t in range(len(d)):
        L = (1 - k) * L + d[t] + np.sqrt(2) * sigma * epsilon[t]
    return L


def confirmation_weighted_integrator(d, epsilon, sigma, phi):
    """
    Glickman, Moran & Usher (2022) consistency-bias integrator.

    Base model is the *perfect* integrator (no leak): X(t) = X(t-1) + mu(t) + eps(t).
    The only deviation from perfect integration is in mu(t):

        mu(t) = d(t) + sign(d(t)) * phi * i

    where i is the length of the run of consecutive same-sign samples
    immediately preceding t (i.e. how many prior frames in a row agree
    with the current frame's sign). i resets to 0 the instant the sign
    flips, and i=0 for the first frame (no preceding run exists yet).

    This is causal: i depends only on d[0..t-1], never looks ahead.

    d       : array of shape (n_frames,) — evidence each frame
    epsilon : array of shape (n_frames,) — shared internal noise
    sigma   : float — internal noise scale
    phi     : float — per-streak-frame boost (free parameter, paper's phi)
    returns : final belief L
    """
    L = 0.0
    streak = 0
    prev_sign = None

    for t in range(len(d)):
        cur_sign = np.sign(d[t])

        # does this frame continue the preceding run?
        if t > 0 and cur_sign != 0 and cur_sign == prev_sign:
            streak += 1
        else:
            streak = 0

        mu = d[t] + cur_sign * phi * streak
        L = L + mu + np.sqrt(2) * sigma * epsilon[t]

        prev_sign = cur_sign

    return L

# ---------------------------------------------------------------------------
# Normative integrator (Glaze, Kable & Gold, 2015)
# ---------------------------------------------------------------------------

def psi(L, H):
    """
    Glaze et al. (2015) swap-aware belief-update nonlinearity.

    Given the previous log-odds belief L and the hazard rate H (probability
    the environment switches between frames), returns the updated log-odds
    belief BEFORE this frame's new evidence is incorporated — i.e. it
    accounts for the possibility that the correct source secretly switched
    since the last observation.

    Derived directly from the predict step in probability space:
        q1_prior = (1-H)*q1 + H*q2
        q2_prior = H*q1 + (1-H)*q2
        psi = log(q1_prior / q2_prior)
    rewritten as a function of L = log(q1/q2) alone. Verified numerically
    against the direct predict-step computation, including two special
    cases:
        H = 0    -> psi(L, 0)   = L   (no switch risk, perfect carry-over)
        H = 0.5  -> psi(L, 0.5) = 0   (total memory wipe every frame)

    L : float — previous log-odds belief, log(q(Left high)/q(Right high))
    H : float — hazard rate (probability of a switch at this step)
    returns : updated log-odds belief after accounting for switch risk
    """
    return np.log((1 - H) + H * np.exp(-L)) - np.log(H + (1 - H) * np.exp(-L))


def normative_integrator(d, epsilon, sigma, H,
                          mu_high=MU_HIGH, mu_low=MU_LOW, sigma_m=SIGMA_M):
    """
    Rename to Glaze Model
    Glaze, Kable & Gold (2015) normative integrator.

    Each frame:
      1. predict — apply psi(L, H) to carry the previous belief forward,
         accounting for the chance the environment switched. Skipped on
         the first frame (equivalent to psi(0, H) = 0 for any H anyway).
      2. update  — corrupt this frame's evidence d[t] with internal noise
         (encoding-noise formulation, same convention as perfect_integrator
         and leaky_integrator), convert to a log-likelihood ratio (LLR),
         and add it to the belief.

    Internal noise is injected into the evidence itself, not the belief:
    this represents noisy sensory encoding of the payout difference each
    frame, applied before the LLR is computed, rather than noise in the
    stored belief.

    d       : array of shape (n_frames,) — evidence each frame (x_left - x_right)
    epsilon : array of shape (n_frames,) — shared internal noise draws ~ N(0,1)
    sigma   : float — internal noise scale (same units/scale as sigma_m)
    H       : float — hazard rate used by the model. Typically the empirical
              switch rate realized in this specific trial, rather than the
              nominal generative H, so the model is given the actual realized
              volatility of the sequence it is integrating.
    mu_high : float — mean of high distribution
    mu_low  : float — mean of low distribution
    sigma_m : float — std of each machine
    returns : final belief L(12), log-odds that Left is high
    """
    coef = (mu_high - mu_low) / sigma_m ** 2
    L = 0.0

    for t in range(len(d)):
        if t > 0:
            L = psi(L, H)

        d_noisy = d[t] + np.sqrt(2) * sigma * epsilon[t]
        L = L + coef * d_noisy

    return L