"""
Central location for all parameters used across:
  - sequence_generation.py            (generating trial sequences)
  - integrators.py                    (the belief-update models)
  - 01_integrators_simulation.ipynb   (sweeping sigma x H and comparing models)
  - 02_parameter_recovery.ipynb

Edit values here only. Every other file imports from this module so there is
a single source of truth instead of the same constants being duplicated (and
potentially drifting out of sync) across files.
"""

# ---------------------------------------------------------------------------
# Task / generative parameters
# ---------------------------------------------------------------------------
MU_HIGH = 53   # mean payout of the "high" machine
MU_LOW  = 47   # mean payout of the "low" machine
SIGMA_M = 15   # machine noise (std of each slot machine's payout)

N_FRAMES = 12  # number of frames (observations) per trial

# ---------------------------------------------------------------------------
# Simulation sweep parameters (used by 01_integrators_simulation.ipynb)
# ---------------------------------------------------------------------------
N_TRIALS     = 1000                 # trials simulated per (sigma, H) condition
H_VALUES     = [0, 0.1, 0.2, 0.3]   # nominal swap-probability values to sweep
SIGMA_VALUES = [0, 10, 20, 30, 40, 50]          # internal noise levels to sweep

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
K = 0.2   # default leak rate for leaky_integrator / confirmation_weighted_integrator

# ---------------------------------------------------------------------------
# Standalone sequence export parameters (used by sequence_generation.py)
# ---------------------------------------------------------------------------
N_SEQUENCES = 5           # how many sequences (rows) to generate for the CSV
H           = 0.1         # swap probability per frame for the exported CSV
SEED        = 42          # random seed for reproducibility
OUTPUT_FILE = "sequences.csv"
