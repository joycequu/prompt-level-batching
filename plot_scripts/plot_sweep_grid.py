"""
Plot a 5×5 sweep grid: dataset (rows) × model (cols) — Total Latency vs Batch Size.
No regression overlay. Axes are independently scaled per cell; legend is shared.

Usage
-----
python plot_scripts/plot_sweep_grid.py --config plot_scripts/sweep_config.json --beta-values

python plot_scripts/plot_sweep_grid.py \
    --config plot_scripts/sweep_config.json \
    --output plots/sweep_grid.pdf

Config JSON format
------------------
{
    "row_labels": [
        "TweetEval Irony", "TweetEval Emotion",
        "SemBench", "LitSearch", "CUAD"
    ],
    "col_labels": [
        "Gemini-3-Flash", "GPT-5.4", "GPT-5.4 Nano",
        "Haiku 4.5", "Sonnet 4.6"
    ],
    "metric_col":   "accuracy",
    "metric_label": "Accuracy",
    "cells": {
        "0,0": {"results_dir": "tweeteval/results/irony_gemini"},
        "0,1": {
            "results_dir": "tweeteval/results/irony_gpt54",
            "metric_col":   "f1",
            "metric_label": "F1"
        }
    }
}

Cell keys are "row,col" (0-indexed). Omitted or missing cells render as "N/A".
Paths under results_dir are relative to --data-folder (or CWD when omitted).
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "legend.fontsize": 11,
    }
)

FS_CELL_TICK = 10  # axis tick labels inside each cell
FS_CELL_LABEL = 11  # axis titles inside each cell
FS_COL_TITLE = 14  # column header (model name)
FS_ROW_LABEL = 14  # row label (dataset name)
FS_LEGEND = 14  # shared legend

from plot_latency_cost_batch_size import (
    _find_per_call_csv,
    _filter_run,
    analyze_latency_tokens,
    save_beta_values,
)


_SWEEP_BETA_CSV = Path(__file__).resolve().parent.parent / "overall_results" / "sweep_beta_values.csv"


def _save_beta_values_for_cell(results_dir: Path, batch_sizes: list | None = None) -> None:
    """Fit the latency regression for one cell and upsert its beta values into the sweep CSV."""
    try:
        regression_stats = analyze_latency_tokens(
            str(results_dir), use_nnls=True, batch_sizes=batch_sizes
        )
        save_beta_values(str(results_dir), regression_stats, csv_path=_SWEEP_BETA_CSV)
    except Exception as e:
        print(f"    beta values skipped: {e}")


def _bs_filter(results_dir: Path, cc: dict) -> list | None:
    """Return the batch-size filter for a cell from config, or None to use all."""
    bs = cc.get("batch_sizes")
    return bs if bs else None


C_TTFT = "#aec6f7"
C_DECODE = "#1f6fbf"
C_LATENCY = "tab:blue"

# One distinct color per quality metric so they're consistent across all cells.
METRIC_STYLE: dict[str, dict] = {
    "accuracy": {"color": "#2ca02c", "marker": "s", "label": "Accuracy"},
    "f1": {"color": "#ff7f0e", "marker": "^", "label": "F1"},
    "precision": {"color": "#9467bd", "marker": "D", "label": "Precision"},
    "recall": {"color": "#d62728", "marker": "v", "label": "Recall"},
}
# Only show the two primary quality metrics; precision/recall are omitted to avoid clutter.
METRIC_PRIORITY = ["accuracy", "f1"]


def _load_cell(
    results_dir: Path,
    metric_col: str,
    run_id: int = 1,
    batch_sizes: list | None = None,
) -> dict | None:
    """Return plot-ready data dict, or None if the results directory is unusable."""
    summary_csv = results_dir / "batch_size_summary.csv"
    if not summary_csv.exists():
        return None

    # FIX 1: Skip bad lines and filter by run_id just like the base script
    df = pd.read_csv(summary_csv, on_bad_lines="skip")
    df = _filter_run(df, run_id).sort_values("batch_size")
    if batch_sizes is not None:
        df = df[df["batch_size"].isin(batch_sizes)].reset_index(drop=True)

    if "total_latency_secs" not in df.columns or df.empty:
        return None

    xs = df["batch_size"].tolist()
    total_lat = df["total_latency_secs"].tolist()

    ttft_totals, decode_totals = [], []
    for bs in xs:
        p = _find_per_call_csv(results_dir, bs)
        if p and p.exists():
            # FIX 2: Filter the per-call data by run_id before summing
            pcd = pd.read_csv(p, on_bad_lines="skip")
            pcd = _filter_run(pcd, run_id)

            if "ttft_secs" in pcd.columns and pcd["ttft_secs"].notna().any():
                ttft_totals.append(float(pcd["ttft_secs"].sum()))
                decode_totals.append(
                    float((pcd["latency_secs"] - pcd["ttft_secs"]).sum())
                )
                continue
        ttft_totals.append(None)
        decode_totals.append(None)

    # Load only the requested metric; fall back to other METRIC_PRIORITY cols if absent.
    metrics: dict[str, list] = {}
    for col in [metric_col] + [c for c in METRIC_PRIORITY if c != metric_col]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").tolist()
            if any(v == v for v in vals):  # at least one non-NaN
                metrics[col] = vals
                break

    return {
        "xs": xs,
        "total_lat": total_lat,
        "ttft": ttft_totals,
        "decode": decode_totals,
        "metrics": metrics,
    }


def _draw_cell(
    ax,
    data: dict,
    *,
    show_xlabel: bool,
    show_left_ylabel: bool,
    show_right_ylabel: bool,
    x_tick_labels: set | None = None,
) -> dict:
    """Draw one latency panel onto ax. Returns a dict of legend handles."""
    xs = data["xs"]
    handles = {}

    has_decomp = any(v is not None for v in data["ttft"])
    if has_decomp:
        import numpy as np

        # Convert to floats and keep None as NaN so matplotlib skips plotting them
        ttft_p = np.array([float(v) if v is not None else np.nan for v in data["ttft"]])
        dec_p = np.array(
            [float(v) if v is not None else np.nan for v in data["decode"]]
        )

        # Calculate total stack height safely handling NaNs
        total_stack = np.nansum([ttft_p, dec_p], axis=0)
        # Handle cases where all values are NaN
        total_stack[np.isnan(ttft_p) & np.isnan(dec_p)] = np.nan

        # Use fill_between with where clauses to completely ignore missing x=1 data
        mask = ~np.isnan(ttft_p)
        ax.fill_between(
            xs, 0, ttft_p, where=mask, facecolor=C_TTFT, alpha=0.55, label="Total TTFT"
        )
        ax.fill_between(
            xs,
            ttft_p,
            total_stack,
            where=mask,
            facecolor=C_DECODE,
            alpha=0.55,
            label="Total Decoding",
        )

        handles["ttft"] = mpatches.Patch(
            facecolor=C_TTFT, alpha=0.55, label="Total TTFT"
        )
        handles["decode"] = mpatches.Patch(
            facecolor=C_DECODE, alpha=0.55, label="Total Decoding"
        )

    (line_lat,) = ax.plot(
        xs,
        data["total_lat"],
        color=C_LATENCY,
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Total Latency",
    )
    handles["latency"] = line_lat
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.5)
    show_ticks = [b for b in xs if x_tick_labels is None or b in x_tick_labels]
    ax.set_xticks(show_ticks)
    ax.set_xticklabels(
        [str(b) for b in show_ticks],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=FS_CELL_TICK,
    )
    ax.tick_params(axis="y", labelsize=FS_CELL_TICK, labelcolor=C_LATENCY)
    if show_xlabel:
        ax.set_xlabel("Batch Size", fontsize=FS_CELL_LABEL)
    if show_left_ylabel:
        ax.set_ylabel("Total Latency (s)", fontsize=FS_CELL_LABEL, color=C_LATENCY)

    if data["metrics"]:
        ax_m = ax.twinx()
        ax_m.set_ylim(0, 1.05)
        for col, vals in data["metrics"].items():
            style = METRIC_STYLE.get(
                col, {"color": "gray", "marker": "o", "label": col}
            )
            (line,) = ax_m.plot(
                xs,
                vals,
                color=style["color"],
                marker=style["marker"],
                markersize=4,
                linewidth=1.5,
                label=style["label"],
                zorder=3,
            )
            handles[col] = line
        if show_right_ylabel:
            label_parts = "/".join(
                METRIC_STYLE.get(c, {}).get("label", c) for c in data["metrics"]
            )
            first_col = next(iter(data["metrics"]))
            metric_color = METRIC_STYLE.get(first_col, {}).get("color", "gray")
            ax_m.set_ylabel(label_parts, fontsize=FS_CELL_LABEL, color=metric_color)
            ax_m.tick_params(axis="y", labelsize=FS_CELL_TICK, labelcolor=metric_color)
        else:
            ax_m.set_yticklabels([])
            ax_m.tick_params(axis="y", length=0)
            ax_m.spines["right"].set_visible(False)

    return handles


def plot_sweep_grid(
    config_path: str,
    data_folder: str | None = None,
    output: str | None = None,
    x_tick_labels: set | None = None,
    beta_values: bool = False,
):
    p = Path(config_path)
    if not p.exists():
        # Try resolving relative to the script's directory (plot_scripts/)
        p = Path(__file__).resolve().parent / config_path
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)

    row_labels = cfg["row_labels"]
    col_labels = cfg["col_labels"]
    default_mcol = cfg.get("metric_col", "accuracy")
    cells_cfg = cfg.get("cells", {})
    if x_tick_labels is None and "x_tick_labels" in cfg:
        x_tick_labels = set(cfg["x_tick_labels"])
    nrows = len(row_labels)
    ncols = len(col_labels)
    base = Path(data_folder) if data_folder else Path(".")

    # --beta-values: recompute beta values for every cell
    if beta_values:
        print("=== --beta-values: recomputing beta values ===")
        for r in range(nrows):
            for c in range(ncols):
                k = f"{r},{c}"
                cc = cells_cfg.get(k, {})
                if not cc.get("results_dir"):
                    continue
                rd = Path(cc["results_dir"])
                if not rd.is_absolute():
                    rd = base / rd
                bs_filter = _bs_filter(None, cc)
                desc = f"{row_labels[r]}, {col_labels[c]}"
                print(f"\n[{r},{c}] {desc}  →  {rd}")
                _save_beta_values_for_cell(rd, batch_sizes=bs_filter)
        print("\n=== --beta-values complete ===\n")

    # Pre-load all cell data
    cell_data: dict[str, dict | None] = {}
    for r in range(nrows):
        for c in range(ncols):
            k = f"{r},{c}"
            cc = cells_cfg.get(k, {})
            if not cc.get("results_dir"):
                continue
            rd = Path(cc["results_dir"])
            if not rd.is_absolute():
                rd = base / rd
            mcol = cc.get("metric_col", default_mcol)
            bs_filter = _bs_filter(None, cc)
            try:
                cell_data[k] = _load_cell(rd, mcol, batch_sizes=bs_filter)
                if cell_data[k] is None:
                    print(f"  [{r},{c}] no usable data in {rd}")
            except Exception as e:
                print(f"  [{r},{c}] load error: {e}")
                continue

            if cc.get("regression") and cell_data.get(k) is not None:
                print(f"  [{r},{c}] saving beta values → {_SWEEP_BETA_CSV}")
                _save_beta_values_for_cell(rd, batch_sizes=bs_filter)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.5 * ncols, 2.2 * nrows),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.55, wspace=0.25)

    all_handles: dict[str, object] = {}

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            k = f"{r},{c}"
            data = cell_data.get(k)
            cc = cells_cfg.get(k, {})
            bs_filter = _bs_filter(None, cc)
            if "x_tick_labels" in cc:
                cell_x_ticks = set(cc["x_tick_labels"])
            elif bs_filter is not None:
                cell_x_ticks = set(bs_filter)
            else:
                cell_x_ticks = x_tick_labels

            if data is None:
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#aaaaaa",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color("#dddddd")
            else:
                h = _draw_cell(
                    ax,
                    data,
                    show_xlabel=(r == nrows - 1),
                    show_left_ylabel=(c == 0),
                    show_right_ylabel=(c == ncols - 1),
                    x_tick_labels=cell_x_ticks,
                )
                all_handles.update(h)

    # Column headers (model names)
    for c, lbl in enumerate(col_labels):
        axes[0][c].set_title(lbl, fontsize=FS_COL_TITLE, fontweight="bold", pad=10)

    # Row labels — wrapped at ~13 chars so long names span two lines
    for r, lbl in enumerate(row_labels):
        wrapped = textwrap.fill(lbl, width=13)
        axes[r][0].text(
            -0.35,
            0.5,
            wrapped,
            transform=axes[r][0].transAxes,
            ha="right",
            va="center",
            fontsize=FS_ROW_LABEL,
            fontweight="bold",
            clip_on=False,
            linespacing=1.3,
        )

    # Shared legend — latency items first, then one entry per quality metric
    latency_order = ["ttft", "decode", "latency"]
    metric_order = [k for k in METRIC_PRIORITY if k in all_handles]
    legend_handles = [
        all_handles[k] for k in latency_order + metric_order if k in all_handles
    ]
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(legend_handles),
            frameon=False,
            fontsize=FS_LEGEND,
        )
        fig.subplots_adjust(bottom=0.101)

    out_path = Path(output) if output else Path("plots/sweep_grid.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="5×5 sweep grid of total latency vs batch size (dataset × model)."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to sweep config JSON.",
    )
    parser.add_argument(
        "--data-folder",
        default=None,
        dest="data_folder",
        help="Base folder; relative results_dir paths resolve against this.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path (default: plots/sweep_grid.pdf).",
    )
    parser.add_argument(
        "--x-ticks",
        nargs="+",
        type=int,
        default=None,
        dest="x_ticks",
        help="Batch sizes to label on the x-axis (e.g. --x-ticks 1 10 100 500). "
        "All ticks are shown; only these get labels.",
    )
    parser.add_argument(
        "--beta-values",
        action="store_true",
        default=False,
        dest="beta_values",
        help="Recompute the NNLS latency-regression beta values for every cell.",
    )
    args = parser.parse_args()
    x_tick_labels = set(args.x_ticks) if args.x_ticks else None
    plot_sweep_grid(
        args.config, args.data_folder, args.output, x_tick_labels,
        beta_values=args.beta_values,
    )


if __name__ == "__main__":
    main()
