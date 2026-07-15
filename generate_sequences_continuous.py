"""
generate_sequences_continous.py
─────────────────────────────────────────────────────────────────────────────
Generates sequences.csv for the belief-tracking Distribution Judgment Task.
Each row is one trial with N_FRAMES = 30 paired (value, switched) tuples.

Usage:  python generate_sequences.py
"""

import numpy as np
import pandas as pd

# ─── Parameters ───────────────────────────────────────────────────────────────
MU_HIGH      = 53
MU_LOW       = 47
SIGMA_M      = 15       # within-box noise (std)
N_FRAMES     = 30       # frames per trial (not revealed to participant)
N_SEQUENCES  = 20       # how many trials to generate
H            = 0.1      # swap probability per frame (starting frame 2)
SEED         = 42
OUTPUT_FILE  = "sequences.csv"

# ─── Generate ─────────────────────────────────────────────────────────────────
rng  = np.random.default_rng(SEED)
rows = []

for seq in range(N_SEQUENCES):
    left_is_high  = True
    tuples_left   = []
    tuples_right  = []

    for t in range(N_FRAMES):
        switched = False
        if t > 0 and rng.random() < H:
            left_is_high = not left_is_high
            switched     = True

        mu_left  = MU_HIGH if left_is_high else MU_LOW
        mu_right = MU_LOW  if left_is_high else MU_HIGH

        sample_l = rng.normal(mu_left,  SIGMA_M)
        sample_r = rng.normal(mu_right, SIGMA_M)

        tuples_left.append( (round(sample_l, 4), int(switched)) )
        tuples_right.append((round(sample_r, 4), int(switched)) )

    final_high_side = "left" if left_is_high else "right"

    # Pre-generate payouts for both possible final choices.
    # If left is high at end: choosing left draws from MU_HIGH, right from MU_LOW.
    mu_left_end  = MU_HIGH if left_is_high else MU_LOW
    mu_right_end = MU_LOW  if left_is_high else MU_HIGH
    payout_if_left  = round(rng.normal(mu_left_end,  SIGMA_M), 4)
    payout_if_right = round(rng.normal(mu_right_end, SIGMA_M), 4)

    row = {f"left_{t+1}":  tuples_left[t]  for t in range(N_FRAMES)}
    row.update({f"right_{t+1}": tuples_right[t] for t in range(N_FRAMES)})
    row["high_side_final"]  = final_high_side
    row["payout_if_left"]   = payout_if_left
    row["payout_if_right"]  = payout_if_right
    rows.append(row)

# ─── Save ─────────────────────────────────────────────────────────────────────
left_cols  = [f"left_{t+1}"  for t in range(N_FRAMES)]
right_cols = [f"right_{t+1}" for t in range(N_FRAMES)]
df = pd.DataFrame(rows, columns=left_cols + right_cols + ["high_side_final", "payout_if_left", "payout_if_right"])
df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {N_SEQUENCES} sequences → '{OUTPUT_FILE}'")
print(f"Parameters: H={H}, MU_HIGH={MU_HIGH}, MU_LOW={MU_LOW}, "
      f"SIGMA_M={SIGMA_M}, N_FRAMES={N_FRAMES}, SEED={SEED}")
print(f"Shape: {df.shape}")
print(df.head(3).to_string())