import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MU_HIGH     = 53
MU_LOW      = 47
SIGMA_M     = 8     # machine noise (std of each slot machine)
N_FRAMES    = 12

N_TRIALS_PER_H = 25
H_VALUES       = [0, 0.1, 0.3]   # swap probability per frame (starting frame 2)
LAST_SWITCH_FRAME = 10      # no switches after this frame (1-indexed, inclusive)

SEED        = 42    # random seed for reproducibility

OUTPUT_FILE = "sequences.csv"

PRACTICE_H_VALUES = [0, 0.1, 0.3, 0.1, 0.3]  # one practice trial per entry
PRACTICE_FILE     = "practice_sequences.csv"

# ---------------------------------------------------------------------------
# Generate one trial's sequence for a given hazard rate h
# ---------------------------------------------------------------------------
def generate_trial(rng, h):
    left_is_high = rng.random() < 0.5  # randomize starting side, independent of h
    tuples_left  = []
    tuples_right = []

    for t in range(N_FRAMES):
        switched = False
        if 0 < t < LAST_SWITCH_FRAME and rng.random() < h:
            left_is_high = not left_is_high
            switched = True

        mu_left  = MU_HIGH if left_is_high else MU_LOW
        mu_right = MU_LOW  if left_is_high else MU_HIGH
        
        sample_l = rng.normal(mu_left,  SIGMA_M)
        sample_r = rng.normal(mu_right, SIGMA_M)

        # Round to nearest integer, clip at 0
        sample_l = max(0, int(round(sample_l)))
        sample_r = max(0, int(round(sample_r)))

        tuples_left.append((sample_l, int(switched)))
        tuples_right.append((sample_r, int(switched)))

    final_high_side = "left" if left_is_high else "right"
    return tuples_left, tuples_right, final_high_side


# ---------------------------------------------------------------------------
# Generate all trials: N_TRIALS_PER_H at each H in H_VALUES
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
rows = []

for h in H_VALUES:
    for _ in range(N_TRIALS_PER_H):
        tuples_left, tuples_right, final_high_side = generate_trial(rng, h)

        row = {f"left_{t+1}": tuples_left[t] for t in range(N_FRAMES)}
        row.update({f"right_{t+1}": tuples_right[t] for t in range(N_FRAMES)})
        row["H"] = h
        row["high_side_final"] = final_high_side
        rows.append(row)

# Keep the H conditions as separate blocks, in H_VALUES order
# (no shuffling — all H=0 trials first, then H=0.1, then H=0.3)

# Practice trials (drawn after the main trials, from the same rng)
practice_rows = []
for h in PRACTICE_H_VALUES:
    tuples_left, tuples_right, final_high_side = generate_trial(rng, h)
    row = {f"left_{t+1}": tuples_left[t] for t in range(N_FRAMES)}
    row.update({f"right_{t+1}": tuples_right[t] for t in range(N_FRAMES)})
    row["H"] = h
    row["high_side_final"] = final_high_side
    practice_rows.append(row)

# ---------------------------------------------------------------------------
# Save to CSV
# ---------------------------------------------------------------------------
left_cols  = [f"left_{t+1}"  for t in range(N_FRAMES)]
right_cols = [f"right_{t+1}" for t in range(N_FRAMES)]
df = pd.DataFrame(rows, columns=left_cols + right_cols + ["H", "high_side_final"])

df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(df)} trials to '{OUTPUT_FILE}'")

practice_df = pd.DataFrame(practice_rows, columns=left_cols + right_cols + ["H", "high_side_final"])
practice_df.to_csv(PRACTICE_FILE, index=False)
print(f"Saved {len(practice_df)} practice trials to '{PRACTICE_FILE}'")
print(f"Parameters: H values={H_VALUES} ({N_TRIALS_PER_H} trials each), "
      f"MU_HIGH={MU_HIGH}, MU_LOW={MU_LOW}, SIGMA_M={SIGMA_M}, seed={SEED}")
print(f"Shape: {df.shape}  ({len(df)} rows x {len(df.columns)} cols)")
print(df.head().to_string())
