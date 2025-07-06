import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets.ostrinia import Ostrinia

# ──────────────────────────────────────────────────────────────────────────────
# Helper: attach the example visualizer if it lives in another file
# (update the import path if you placed the class elsewhere).
# ──────────────────────────────────────────────────────────────────────────────
try:
    from tools.visualize_data import SpatiotemporalVisualizer  # noqa: E402
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Could not import SpatiotemporalVisualizer. "
        "Ensure that the example visualizer class is available as a module "
        "(e.g. spatiotemporal_visualizer.py) in your PYTHONPATH."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Core routine
# ──────────────────────────────────────────────────────────────────────────────
def visualise_years(
    years: List[int],
    target: str = "incrementing_ostrinia",
    root: str = "datasets",
    output_dir: str = "plots",
    line_width: float = 1.0,
    line_alpha: float = 0.9,
    show: bool = False,
    ds: Ostrinia = None,
) -> None:
    """
    Generate one PNG per node whose *clean* signal is not all-zero across the
    selected calendar years.  Each figure shows:

        • the node’s clean signal (primary y-axis)
        • that node’s own increment_flag (secondary y-axis)

    Output files:  <output_dir>/<target>_<minYear>–<maxYear>/node_<node>.png
    """
    # 1. Load dataset
    # ds = Ostrinia(root=root, target=target, smooth=False, drop_nodes=True)
    df: pd.DataFrame = ds.target.copy()               # MultiIndex columns
    df.index = pd.to_datetime(df.index)

    # 2. Restrict to requested years
    year_mask = df.index.year.isin(years)
    if not year_mask.any():
        raise ValueError(f"No observations fall in the years {years}.")
    df_sel = df.loc[year_mask]

    # --- split signals and flags ---------------------------------------
    # Assumes column MultiIndex = (node, channel)
    df_clean  = df_sel.xs("clean",     level="channels", axis=1)
    df_incflt = df_sel.xs("increment", level="channels", axis=1)

    # 3. Prepare output directory
    label   = f"{target}_{min(years)}–{max(years)}"
    out_dir = os.path.join(output_dir, label)
    os.makedirs(out_dir, exist_ok=True)

    # 4. Per-node plotting
    for node in df_clean.columns:
        sig = df_clean[node]

        # Skip nodes with a flat-zero signal
        if np.allclose(sig.values, 0):
            print(f"⤻  node {node} skipped (signal all zeros)")
            continue

        # Retrieve this node’s increment flag; if absent, skip gracefully
        try:
            flag = df_incflt[node].astype(int)
        except KeyError:
            print(f"⤻  node {node} skipped (no increment_flag for this node)")
            continue

        # Figure
        fig, ax = plt.subplots(figsize=(12, 4))

        # signal
        ax.plot(sig.index, sig.values, lw=line_width, alpha=line_alpha)
        ax.set_ylabel(f"signal – node {node}")

        # flag on twin y-axis
        ax2 = ax.twinx()
        ax2.step(flag.index, flag.values, where="post", lw=1.5, alpha=0.75)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["off", "on"])
        ax2.set_ylabel("increment_flag")

        # cosmetics
        ax.set_xlabel("Date")
        ax.set_title(f"{label} – node {node}")
        ax.grid(True, alpha=0.3)

        # save / show
        fname = f"node_{node}.png"
        path  = os.path.join(out_dir, fname)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        if show:
            plt.show()
        plt.close(fig)

        print(f"✓ {fname} saved → {path}")

    print("All requested per-node plots have been generated.")

# ──────────────────────────────────────────────────────────────────────────────
# Command‑line entry point for reproducibility from shell / scripts
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Example of using with your own data:

    years = [2022, 2023]  # Example years to visualize
    
    # nb_ostrinia, incrementing_ostrinia
    target = 'incrementing_ostrinia'
    # target = "nb_ostrinia"

    outputdir = "plots/" + target

    dataset = Ostrinia(root="datasets", target=target, smooth=False, drop_nodes=True)

    visualise_years(years, target=target, root="datasets", output_dir=outputdir, ds=dataset)
