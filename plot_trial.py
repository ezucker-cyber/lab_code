"""
plot_trial.py
─────────────────────────────────────────────────────────────────────────────
Plots a single trial from the Distribution Judgment (belief-tracking) task.

For each trial it shows:
  • Left axis  — the raw sample values drawn from Left and Right boxes over
                 time, with the running true-mean lines overlaid.
  • Right axis — the participant's recorded belief (P(Right)) after each frame,
                 as a step-function in 10 % increments.
  • Vertical dashed lines wherever a change-point occurred (switched == 1).

Usage
─────
    python plot_trial.py \
        --results results_P001.csv \
        --sequences sequences.csv \
        --trial 1              # 1-indexed
        [--out trial1.png]

If --out is omitted the plot is shown interactively.

Dependencies: numpy, pandas, matplotlib
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─── Parameters (must match generate_sequences.py) ────────────────────────────
MU_HIGH  = 53
MU_LOW   = 47
N_FRAMES = 30


# ─── Helpers ──────────────────────────────────────────────────────────────────
def parse_tuple(s: str) -> dict:
    """Parse '(value, switched)' string from CSV into dict."""
    clean = s.replace("(", "").replace(")", "").strip()
    parts = clean.split(",")
    return {"value": float(parts[0]), "switched": int(parts[1])}


def load_trial_data(results_path: str, seq_path: str, trial_num: int):
    """
    Returns
    -------
    frames : list of dicts  [{left_val, left_sw, right_val, right_sw}, ...]
    beliefs: list of floats  length = n_frames  (participant's P(Right) 0–100)
    high_side : 'left' | 'right'
    final_belief : float
    chosen_side  : str
    correct      : int  (0 or 1)
    """
    results_df = pd.read_csv(results_path)
    seq_df     = pd.read_csv(seq_path)

    # results are 1-indexed in the trial column
    r_row = results_df[results_df["trial"] == trial_num]
    if r_row.empty:
        sys.exit(f"Trial {trial_num} not found in {results_path}.")
    r_row = r_row.iloc[0]

    # The sequence file is 0-indexed (row 0 = trial 1)
    s_row = seq_df.iloc[trial_num - 1]

    frames  = []
    beliefs = []
    for t in range(1, N_FRAMES + 1):
        lkey = f"left_{t}"
        rkey = f"right_{t}"
        bkey = f"belief_after_{t}"

        # Some sequences may be shorter if CSV was generated with fewer frames
        if lkey not in s_row or pd.isna(s_row[lkey]) or s_row[lkey] == "":
            break

        lp = parse_tuple(str(s_row[lkey]))
        rp = parse_tuple(str(s_row[rkey]))
        frames.append({
            "left_val":  lp["value"],
            "left_sw":   lp["switched"],
            "right_val": rp["value"],
            "right_sw":  rp["switched"],
        })

        bval = r_row.get(bkey, np.nan)
        beliefs.append(float(bval) if bval != "" else np.nan)

    return (
        frames,
        beliefs,
        str(r_row["high_side_final"]),
        float(r_row["final_belief"]),
        str(r_row["chosen_side"]),
        int(r_row["correct"]),
    )


# ─── Plot ─────────────────────────────────────────────────────────────────────
def plot_trial(frames, beliefs, high_side, final_belief, chosen_side, correct,
               trial_num, out_path=None):

    n  = len(frames)
    xs = list(range(1, n + 1))  # 1-indexed

    left_vals  = [f["left_val"]  for f in frames]
    right_vals = [f["right_val"] for f in frames]

    # Cumulative running mean of actual samples drawn
    left_running  = [sum(left_vals[:i+1])  / (i+1) for i in range(n)]
    right_running = [sum(right_vals[:i+1]) / (i+1) for i in range(n)]

    # Change-point positions (frames where a switch occurred, 1-indexed)
    change_points = [i + 1 for i, f in enumerate(frames) if f["left_sw"]]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()

    # ── Running sample means (left axis) ─────────────────────────────────────
    ax1.plot(xs, left_running,  color="#4e79a7", linewidth=2.2,
             zorder=4, label="Running mean – Left")
    ax1.plot(xs, right_running, color="#e15759", linewidth=2.2,
             zorder=4, label="Running mean – Right")

    ax1.set_xlabel("Frame", fontsize=11)
    ax1.set_ylabel("Running sample mean", fontsize=11)
    ax1.set_xlim(0.5, n + 0.5)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # ── Belief step trace (right axis) ───────────────────────────────────────
    # Build a proper step: belief recorded after frame t holds until frame t+1
    # Start at 50 before any frame, then step at each frame boundary
    belief_xs = [1] + xs                          # starts at x=1, steps at each frame
    belief_ys = [50.0] + [b if not np.isnan(b) else 50.0 for b in beliefs]

    ax2.step(belief_xs, belief_ys, where="post", color="#59a14f",
             linewidth=2.2, zorder=5, label="P(Right) belief")

    # Extend the final belief to the right edge
    ax2.plot([xs[-1], n + 0.5], [belief_ys[-1], belief_ys[-1]],
             color="#59a14f", linewidth=2.2, zorder=5)

    ax2.set_ylabel("P(Right higher) [%]", fontsize=11, color="#59a14f")
    ax2.tick_params(axis="y", labelcolor="#59a14f")
    ax2.set_ylim(-5, 105)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax2.axhline(50, color="#59a14f", linewidth=0.8, linestyle="--", alpha=0.3, zorder=1)

    # ── Background shading: blue=left higher, red=right higher ──────────────
    segment_starts = [0.5] + [cp - 0.5 for cp in change_points]
    segment_ends   = [cp - 0.5 for cp in change_points] + [n + 0.5]
    seg_left_high  = True   # sequence always starts with left = high
    left_labeled   = False
    right_labeled  = False
    for x0, x1 in zip(segment_starts, segment_ends):
        if seg_left_high:
            lbl = "Left higher" if not left_labeled else None
            ax1.axvspan(x0, x1, color="#4e79a7", alpha=0.08, zorder=0, label=lbl)
            left_labeled = True
        else:
            lbl = "Right higher" if not right_labeled else None
            ax1.axvspan(x0, x1, color="#e15759", alpha=0.08, zorder=0, label=lbl)
            right_labeled = True
        seg_left_high = not seg_left_high

    # ── Change-point verticals ────────────────────────────────────────────────
    for i, cp in enumerate(change_points):
        ax1.axvline(cp - 0.5, color="#f28e2b", linewidth=1.8, linestyle="--",
                    zorder=2, alpha=0.85,
                    label="Change point" if i == 0 else None)

    # ── Title ─────────────────────────────────────────────────────────────────
    outcome_str = "✓ Correct" if correct else "✗ Incorrect"
    fig.suptitle(
        f"Trial {trial_num}  ·  True higher: {high_side.upper()}  ·  "
        f"Chosen: {chosen_side.upper()}  ·  Final belief P(Right)={final_belief:.0f}%  ·  {outcome_str}",
        fontsize=11, fontweight="bold", y=1.01
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles1 + handles2, labels1 + labels2):
        if l not in seen:
            seen[l] = h
    fig.legend(seen.values(), seen.keys(), loc="lower center",
               ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.13),
               frameon=True, framealpha=0.9)

    ax1.grid(axis="y", alpha=0.22, linestyle="--")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()


# ─── Batch mode: plot all trials ──────────────────────────────────────────────
def plot_all_trials(results_path, seq_path, out_dir="."):
    import os
    results_df = pd.read_csv(results_path)
    for _, row in results_df.iterrows():
        t = int(row["trial"])
        try:
            data = load_trial_data(results_path, seq_path, t)
        except SystemExit:
            print(f"Skipping trial {t} (data missing).")
            continue
        out = os.path.join(out_dir, f"trial_{t:03d}.png")
        plot_trial(*data, trial_num=t, out_path=out)


# ─── Config — edit these paths before running ─────────────────────────────────
RESULTS_PATH   = "results_01.csv"   # path to your results CSV
SEQUENCES_PATH = "sequences.csv"      # path to your sequences CSV
TRIAL          = 1                    # trial number to plot (1-indexed); set to None to plot all
OUT_PATH       = None                 # e.g. "trial1.png" to save, or None to display interactively
OUT_DIR        = "."                  # output folder when plotting all trials

# ─── Run ──────────────────────────────────────────────────────────────────────
if TRIAL is not None:
    frames, beliefs, high_side, fb, chosen, correct = load_trial_data(
        RESULTS_PATH, SEQUENCES_PATH, TRIAL)
    plot_trial(frames, beliefs, high_side, fb, chosen, correct,
               trial_num=TRIAL, out_path=OUT_PATH)
else:
    plot_all_trials(RESULTS_PATH, SEQUENCES_PATH, out_dir=OUT_DIR)