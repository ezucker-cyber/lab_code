import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MU_HIGH     = 53
MU_LOW      = 47
SIGMA_M     = 15    # machine noise (std of each slot machine)
N_FRAMES    = 12

N_TRIALS_PER_H = 25
H_VALUES       = [0, 0.1]   # swap probability per frame (starting frame 2)

SEED        = 42    # random seed for reproducibility

OUTPUT_FILE = "sequences.csv"

# ---------------------------------------------------------------------------
# Generate one trial's sequence for a given hazard rate h
# ---------------------------------------------------------------------------
def generate_trial(rng, h):
    left_is_high = rng.random() < 0.5  # randomize starting side, independent of h
    tuples_left  = []
    tuples_right = []

    for t in range(N_FRAMES):
        switched = False
        if t > 0 and rng.random() < h:
            left_is_high = not left_is_high
            switched = True

        mu_left  = MU_HIGH if left_is_high else MU_LOW
        mu_right = MU_LOW  if left_is_high else MU_HIGH

        sample_l = rng.normal(mu_left,  SIGMA_M)
        sample_r = rng.normal(mu_right, SIGMA_M)

        # Each entry: (sample_value, switched_this_frame)
        # switched is always 0 for frame 0 (no swap possible on first frame)
        tuples_left.append((round(sample_l, 4), int(switched)))
        tuples_right.append((round(sample_r, 4), int(switched)))

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

# Keep the two H conditions as separate blocks, in H_VALUES order
# (no shuffling — all H=0 trials first, then all H=0.1 trials)

# ---------------------------------------------------------------------------
# Save to CSV
# ---------------------------------------------------------------------------
left_cols  = [f"left_{t+1}"  for t in range(N_FRAMES)]
right_cols = [f"right_{t+1}" for t in range(N_FRAMES)]
df = pd.DataFrame(rows, columns=left_cols + right_cols + ["H", "high_side_final"])

df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(df)} trials to '{OUTPUT_FILE}'")
print(f"Parameters: H values={H_VALUES} ({N_TRIALS_PER_H} trials each), "
      f"MU_HIGH={MU_HIGH}, MU_LOW={MU_LOW}, SIGMA_M={SIGMA_M}, seed={SEED}")
print(f"Shape: {df.shape}  ({len(df)} rows x {len(df.columns)} cols)")
print(df.head().to_string())
