"""
  Left  — Per-call Median latency  (stacked TTFT + Decoding areas)
  Right — Total latency + accuracy (stacked TTFT + Decoding, regression overlay)

Usage:
    python plot_paper_figure.py \
        --results-dir 1_doc_batching/sembench/results/20_or_gem3flash_op_streaming_none_dot_100 \
        --use-nnls --show-regression
"""

import argparse
import re
import sys
from pathlib import Path
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_latency_cost_batch_size import analyze_latency_tokens

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
    }
)

# ── palette ────────────────────────────────────────────────────────────────────
C_TTFT_FILL = "#c6dbef"  # light blue
C_DECODE_FILL = "#4292c6"  # mid blue
C_LATENCY = "#08519c"  # deep navy
C_REGRESSION = "#d62728"  # vivid red
C_ACCURACY = "#2ca02c"  # green
C_F1 = "#ff7f0e"  # orange
C_SPINE = "#bbbbbb"

FS_LABEL = 9
FS_TICK = 9
FS_LEGEND = 9
FS_TITLE = 10


# ── helpers ────────────────────────────────────────────────────────────────────


def _clean_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_SPINE)
        ax.spines[s].set_linewidth(0.7)
    ax.tick_params(labelsize=FS_TICK, direction="out", length=3, width=0.7)
    ax.grid(
        axis="y", linestyle="--", linewidth=0.35, alpha=0.5, color="#aaaaaa", zorder=0
    )


_PREFERRED_TICKS = {1, 10, 20, 50, 100}


def _setup_x(ax, x_vals, tick_subset=None):
    all_vals = sorted(set(x_vals))
    preferred = tick_subset if tick_subset is not None else _PREFERRED_TICKS
    show = sorted(v for v in all_vals if v in preferred) or all_vals
    ax.set_xticks(show)
    ax.set_xticklabels(
        [str(int(t)) for t in show],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=FS_TICK,
    )
    span = max(x_vals) - min(x_vals)
    pad = span * 0.04 if span > 0 else max(abs(min(x_vals)) * 0.5, 0.5)
    ax.set_xlim(left=min(x_vals) - pad, right=max(x_vals) + pad)


def _per_call_stats(results_dir: Path, df: pd.DataFrame):
    med_ttft, med_decode = [], []
    tot_ttft, tot_decode = [], []
    for bs in df["batch_size"]:
        candidates = list(results_dir.glob(f"*batch_size_{bs}_per_call.csv"))
        if candidates:
            pc = pd.read_csv(candidates[0])
            ttft_all = pc["ttft_secs"].fillna(0)
            decode_all = (pc["latency_secs"] - ttft_all).clip(lower=0)
            tot_ttft.append(ttft_all.sum())
            tot_decode.append(decode_all.sum())

            docs_col = next(
                (c for c in pc.columns if re.match(r"num_\w+_in_batch$", c)), None
            )
            pc_med = pc
            if docs_col is not None:
                complete = pc[pd.to_numeric(pc[docs_col], errors="coerce") == bs]
                if not complete.empty:
                    pc_med = complete
            ttft_med = pc_med["ttft_secs"].fillna(0)
            decode_med = (pc_med["latency_secs"] - ttft_med).clip(lower=0)
            med_ttft.append(ttft_med.median())
            med_decode.append(decode_med.median())
        else:
            med_ttft.append(np.nan)
            med_decode.append(np.nan)
            tot_ttft.append(np.nan)
            tot_decode.append(np.nan)
    return (
        np.array(med_ttft),
        np.array(med_decode),
        np.array(tot_ttft),
        np.array(tot_decode),
    )


# ── main ───────────────────────────────────────────────────────────────────────


def plot_paper_figure(
    results_dir,
    output_path=None,
    use_nnls=True,
    show_regression=True,
    right_only=False,
    batch_sizes=None,
):
    results_dir = Path(results_dir)
    df = pd.read_csv(results_dir / "batch_size_summary.csv").sort_values("batch_size")
    if batch_sizes is not None:
        df = df[df["batch_size"].isin(batch_sizes)].reset_index(drop=True)
    x = np.array(df["batch_size"].tolist(), dtype=float)

    actual_total = df["total_latency_secs"].values

    if show_regression:
        reg = analyze_latency_tokens(results_dir, use_nnls=use_nnls, batch_sizes=batch_sizes)
        b0, b1, b2 = reg["beta_0"], reg["beta_1"], reg["beta_2"]
        pred_total = (
            df["total_llm_calls"].values * b0
            + df["total_prompt_tokens"].values * b1
            + df["total_completion_tokens"].values * b2
        )
    else:
        pred_total = np.zeros_like(actual_total)

    med_ttft, med_decode, tot_ttft, tot_decode = _per_call_stats(results_dir, df)

    # ── figure layout ──────────────────────────────────────────────────────────
    n_cols = 1 if right_only else 2
    if right_only:
        fig_w, fig_h = 3.4, 2.6
        gs_kw = dict(left=0.10, right=0.90, top=0.78, bottom=0.14)
    else:
        fig_w, fig_h = 5.4, 2.0
        gs_kw = dict(wspace=0.25, left=0.07, right=0.97, top=0.93, bottom=0.22)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(1, n_cols, figure=fig, **gs_kw)

    ax_left = None if right_only else fig.add_subplot(gs[0, 0])
    right_col = 0 if right_only else 1
    ax_rt = fig.add_subplot(gs[0, right_col])

    # ── LEFT panel ────────────────────────────────────────────────────────────
    left_handles = []
    if not right_only:
        _clean_ax(ax_left)
        _setup_x(ax_left, x.tolist())
        ax_left.fill_between(x, 0, med_ttft, color=C_TTFT_FILL, alpha=1.0, zorder=2)
        ax_left.fill_between(
            x,
            med_ttft,
            med_ttft + med_decode,
            color=C_DECODE_FILL,
            alpha=0.85,
            zorder=2,
        )
        ax_left.plot(
            x,
            med_ttft + med_decode,
            color=C_LATENCY,
            marker="o",
            linewidth=1.6,
            markersize=3.5,
            zorder=3,
            label="_nolegend_",
        )
        ax_left.set_ylabel(
            "Per-call Latency (s)",
            fontsize=FS_LABEL,
            color=C_LATENCY,
            labelpad=4,
        )
        ax_left.tick_params(axis="y", labelcolor=C_LATENCY)
        ax_left.set_ylim(bottom=0)
        ax_left.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=False))
        ax_left.set_title(
            "Per-Call Median Latency",
            fontsize=FS_TITLE,
            pad=7,
            fontweight="medium",
        )
        left_handles = [
            Patch(
                facecolor=C_TTFT_FILL, edgecolor=C_SPINE, linewidth=0.5, label="TTFT"
            ),
            Patch(
                facecolor=C_DECODE_FILL,
                edgecolor=C_SPINE,
                linewidth=0.5,
                label="Decoding",
            ),
        ]

    # ── RIGHT panel (Standard single axis) ─────────────────────────────────────
    _clean_ax(ax_rt)
    _setup_x(ax_rt, x.tolist())

    ax_rt.fill_between(x, 0, tot_ttft, color=C_TTFT_FILL, alpha=1.0, zorder=2)
    ax_rt.fill_between(
        x,
        tot_ttft,
        tot_ttft + tot_decode,
        color=C_DECODE_FILL,
        alpha=0.85,
        zorder=2,
    )
    ax_rt.plot(
        x,
        actual_total,
        color=C_LATENCY,
        marker="o",
        linewidth=1.6,
        markersize=3.5,
        zorder=3,
        label="Total latency",
    )
    if show_regression:
        ax_rt.plot(
            x,
            pred_total,
            color=C_REGRESSION,
            linestyle=":",
            linewidth=1.8,
            marker=".",
            markersize=4.5,
            zorder=3,
            label="Regression fit",
        )

    ax_rt.set_ylim(bottom=0)
    ax_rt.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))
    ax_rt.set_xlabel("Batch size", fontsize=FS_LABEL, labelpad=4)
    ax_rt.set_ylabel(
        "Total latency (s)", fontsize=FS_LABEL, color=C_LATENCY, labelpad=4
    )
    ax_rt.tick_params(axis="y", labelcolor=C_LATENCY)

    if not right_only:
        ax_rt.set_title(
            "Total Latency & Correctness",
            fontsize=FS_TITLE,
            pad=7,
            fontweight="medium",
        )

    # Accuracy line on twin axis
    ax_acc = ax_rt.twinx()
    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_color(C_SPINE)
    ax_acc.spines["right"].set_linewidth(0.7)
    ax_acc.plot(
        x,
        df["accuracy"].values,
        color=C_ACCURACY,
        marker="s",
        linestyle="-.",
        linewidth=1.5,
        markersize=3.5,
        zorder=5,
        label="Accuracy",
    )
    ax_acc.set_ylabel("Accuracy", fontsize=FS_LABEL, color=C_ACCURACY, labelpad=4)
    ax_acc.tick_params(axis="y", labelcolor=C_ACCURACY, labelsize=FS_TICK)
    ax_acc.set_ylim(0, 1.12)

    # Setup right panel legend handles
    rp_handles = [
        Patch(facecolor=C_TTFT_FILL, edgecolor=C_SPINE, linewidth=0.5, label="TTFT"),
        Patch(
            facecolor=C_DECODE_FILL, edgecolor=C_SPINE, linewidth=0.5, label="Decoding"
        ),
        plt.Line2D(
            [0],
            [0],
            color=C_LATENCY,
            linewidth=1.4,
            marker="o",
            markersize=3.5,
            label="Total latency",
        ),
    ]
    if show_regression:
        rp_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=C_REGRESSION,
                linewidth=1.4,
                linestyle=":",
                marker=".",
                markersize=4,
                label="Regression fit",
            )
        )
    rp_handles.append(
        plt.Line2D(
            [0],
            [0],
            color=C_ACCURACY,
            linewidth=1.4,
            linestyle="-.",
            marker="s",
            markersize=3.5,
            label="Accuracy",
        )
    )

    # ── unified legend above figure ────────────────────────────────────────────
    _seen = set()
    _legend_handles = []
    for h in left_handles + rp_handles:
        lbl = h.get_label()
        if lbl not in _seen:
            _seen.add(lbl)
            _legend_handles.append(h)

    if right_only:
        fig.legend(
            handles=_legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.80),
            ncol=len(_legend_handles),
            frameon=False,
            handlelength=1.3,
            handletextpad=0.5,
            columnspacing=1.2,
            fontsize=FS_LEGEND,
        )
    else:
        fig.legend(
            handles=_legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=3,
            frameon=False,
            handlelength=1.3,
            handletextpad=0.5,
            columnspacing=2,
            fontsize=FS_LEGEND,
        )

    # ── save ──────────────────────────────────────────────────────────────────
    if output_path is None:
        task_dir = results_dir.parent.parent
        out_dir = task_dir.parent / "plots" / task_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_paper_right" if right_only else "_paper"
        output_path = out_dir / f"{results_dir.name}{suffix}.pdf"

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved → {output_path}")
    plt.show()


def plot_paper_combined(
    sembench_dir,
    cuad_dir,
    output_path=None,
    use_nnls=True,
    show_regression=True,
):
    """Two-row figure: SemBench (top) and CUAD (bottom), two columns each."""

    _CUAD_BS = {1, 2, 5, 10}

    def _load(results_dir, batch_size_filter=None):
        results_dir = Path(results_dir)
        df = pd.read_csv(results_dir / "batch_size_summary.csv").sort_values(
            "batch_size"
        )
        if batch_size_filter is not None:
            df = df[df["batch_size"].isin(batch_size_filter)].reset_index(drop=True)
        x = np.array(df["batch_size"].tolist(), dtype=float)
        actual_total = df["total_latency_secs"].values
        if show_regression:
            # Fit regression on the (possibly filtered) rows so betas reflect this data
            X = np.column_stack(
                [
                    df["total_llm_calls"].values,
                    df["total_prompt_tokens"].values,
                    df["total_completion_tokens"].values,
                ]
            )
            y = df["total_latency_secs"].values
            if use_nnls:
                from scipy.optimize import nnls as _nnls

                b, _ = _nnls(X, y)
            else:
                b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            pred_total = X @ b
        else:
            pred_total = np.zeros_like(actual_total)
        med_ttft, med_decode, tot_ttft, tot_decode = _per_call_stats(results_dir, df)
        return (
            x,
            actual_total,
            pred_total,
            med_ttft,
            med_decode,
            tot_ttft,
            tot_decode,
            df,
        )

    sb_x, sb_at, sb_pt, sb_mt, sb_md, sb_tt, sb_td, sb_df = _load(sembench_dir)
    cu_x, cu_at, cu_pt, cu_mt, cu_md, cu_tt, cu_td, cu_df = _load(
        cuad_dir, batch_size_filter=_CUAD_BS
    )

    fig = plt.figure(figsize=(5.4, 4.2))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.40,
        wspace=0.25,
        left=0.09,
        right=0.97,
        top=0.95,
        bottom=0.20,
    )
    ax_sb_l = fig.add_subplot(gs[0, 0])
    ax_sb_r = fig.add_subplot(gs[0, 1])
    ax_cu_l = fig.add_subplot(gs[1, 0])
    ax_cu_r = fig.add_subplot(gs[1, 1])

    def _draw_left(
        ax, x, med_ttft, med_decode, show_title, show_xlabel, tick_subset=None
    ):
        _clean_ax(ax)
        _setup_x(ax, x.tolist(), tick_subset=tick_subset)
        ax.fill_between(x, 0, med_ttft, color=C_TTFT_FILL, alpha=1.0, zorder=2)
        ax.fill_between(
            x,
            med_ttft,
            med_ttft + med_decode,
            color=C_DECODE_FILL,
            alpha=0.85,
            zorder=2,
        )
        ax.plot(
            x,
            med_ttft + med_decode,
            color=C_LATENCY,
            marker="o",
            linewidth=1.6,
            markersize=3.5,
            zorder=3,
        )
        ax.set_ylabel(
            "Per-call Latency (s)", fontsize=FS_LABEL, color=C_LATENCY, labelpad=4
        )
        ax.tick_params(axis="y", labelcolor=C_LATENCY)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=False))
        if show_title:
            ax.set_title(
                "Per-Call Median Latency", fontsize=FS_TITLE, pad=7, fontweight="medium"
            )
        if show_xlabel:
            ax.set_xlabel("Batch size", fontsize=FS_LABEL, labelpad=4)

    def _draw_right(
        ax,
        x,
        actual_total,
        pred_total,
        tot_ttft,
        tot_decode,
        df_row,
        show_title,
        show_xlabel,
        metric_col="accuracy",
        metric_color=C_ACCURACY,
        metric_marker="s",
        metric_label="Accuracy",
        tick_subset=None,
    ):
        _clean_ax(ax)
        _setup_x(ax, x.tolist(), tick_subset=tick_subset)
        ax.fill_between(x, 0, tot_ttft, color=C_TTFT_FILL, alpha=1.0, zorder=2)
        ax.fill_between(
            x,
            tot_ttft,
            tot_ttft + tot_decode,
            color=C_DECODE_FILL,
            alpha=0.85,
            zorder=2,
        )
        ax.plot(
            x,
            actual_total,
            color=C_LATENCY,
            marker="o",
            linewidth=1.6,
            markersize=3.5,
            zorder=3,
            label="Total latency",
        )
        if show_regression:
            ax.plot(
                x,
                pred_total,
                color=C_REGRESSION,
                linestyle=":",
                linewidth=1.8,
                marker=".",
                markersize=4.5,
                zorder=3,
                label="Regression fit",
            )
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))
        ax.set_ylabel(
            "Total latency (s)", fontsize=FS_LABEL, color=C_LATENCY, labelpad=4
        )
        ax.tick_params(axis="y", labelcolor=C_LATENCY)
        if show_title:
            ax.set_title(
                "Total Latency & Accuracy",
                fontsize=FS_TITLE,
                pad=7,
                fontweight="medium",
            )
        if show_xlabel:
            ax.set_xlabel("Batch size", fontsize=FS_LABEL, labelpad=4)

        ax_m = ax.twinx()
        ax_m.spines["top"].set_visible(False)
        ax_m.spines["right"].set_color(C_SPINE)
        ax_m.spines["right"].set_linewidth(0.7)
        ax_m.plot(
            x,
            df_row[metric_col].values,
            color=metric_color,
            marker=metric_marker,
            linestyle="-.",
            linewidth=1.5,
            markersize=3.5,
            zorder=5,
        )
        ax_m.set_ylabel(metric_label, fontsize=FS_LABEL, color=metric_color, labelpad=4)
        ax_m.tick_params(axis="y", labelcolor=metric_color, labelsize=FS_TICK)
        ax_m.set_ylim(0, 1.12)

    # Top row: SemBench — titles only, no x-axis labels
    _draw_left(ax_sb_l, sb_x, sb_mt, sb_md, show_title=True, show_xlabel=False)
    _draw_right(
        ax_sb_r,
        sb_x,
        sb_at,
        sb_pt,
        sb_tt,
        sb_td,
        sb_df,
        show_title=True,
        show_xlabel=False,
    )

    _CUAD_TICKS = _CUAD_BS  # show all plotted points as ticks
    # Bottom row: CUAD — no titles, x-axis labels, F1 metric
    _draw_left(
        ax_cu_l,
        cu_x,
        cu_mt,
        cu_md,
        show_title=False,
        show_xlabel=True,
        tick_subset=_CUAD_TICKS,
    )
    _draw_right(
        ax_cu_r,
        cu_x,
        cu_at,
        cu_pt,
        cu_tt,
        cu_td,
        cu_df,
        show_title=False,
        show_xlabel=True,
        metric_col="f1",
        metric_color=C_F1,
        metric_marker="^",
        metric_label="F1",
        tick_subset=_CUAD_TICKS,
    )

    # Row labels in the top-left corner of each left panel
    for ax, label in [(ax_sb_l, "SemBench"), (ax_cu_l, "CUAD")]:
        ax.text(
            0.03,
            0.97,
            label,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=FS_TICK,
            fontstyle="italic",
            color="#555555",
        )

    # Shared legend at the bottom
    legend_handles = [
        Patch(facecolor=C_TTFT_FILL, edgecolor=C_SPINE, linewidth=0.5, label="TTFT"),
        Patch(
            facecolor=C_DECODE_FILL, edgecolor=C_SPINE, linewidth=0.5, label="Decoding"
        ),
        plt.Line2D(
            [0],
            [0],
            color=C_LATENCY,
            linewidth=1.4,
            marker="o",
            markersize=3.5,
            label="Total latency",
        ),
    ]
    if show_regression:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=C_REGRESSION,
                linewidth=1.4,
                linestyle=":",
                marker=".",
                markersize=4,
                label="Regression fit",
            )
        )
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            color=C_ACCURACY,
            linewidth=1.4,
            linestyle="-.",
            marker="s",
            markersize=3.5,
            label="Accuracy",
        )
    )
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            color=C_F1,
            linewidth=1.4,
            linestyle="-.",
            marker="^",
            markersize=3.5,
            label="F1",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(legend_handles),
        frameon=False,
        handlelength=1.3,
        handletextpad=0.5,
        columnspacing=1.2,
        fontsize=FS_LEGEND,
    )

    if output_path is None:
        out_dir = Path(sembench_dir).parents[2] / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "combined_paper.pdf"

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved → {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paper-quality latency/accuracy figure(s)"
    )
    # Single-dataset mode
    parser.add_argument("--results-dir", default=None, dest="results_dir")
    # Combined mode
    parser.add_argument("--sembench-dir", default=None, dest="sembench_dir")
    parser.add_argument("--cuad-dir", default=None, dest="cuad_dir")
    # Shared
    parser.add_argument("--output", default=None, dest="output_path")
    parser.add_argument(
        "--use-nnls", action="store_true", default=True, dest="use_nnls"
    )
    parser.add_argument(
        "--show-regression", action="store_true", default=False, dest="show_regression"
    )
    parser.add_argument(
        "--right-only", action="store_true", default=False, dest="right_only"
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=None,
        dest="batch_sizes",
        help="Restrict to these batch sizes for plotting and regression (e.g. --batch-sizes 1 2 5 10).",
    )
    args = parser.parse_args()

    if args.sembench_dir and args.cuad_dir:
        plot_paper_combined(
            args.sembench_dir,
            args.cuad_dir,
            output_path=args.output_path,
            use_nnls=args.use_nnls,
            show_regression=args.show_regression,
        )
    else:
        plot_paper_figure(
            args.results_dir,
            output_path=args.output_path,
            use_nnls=args.use_nnls,
            show_regression=args.show_regression,
            right_only=args.right_only,
            batch_sizes=args.batch_sizes or None,
        )
