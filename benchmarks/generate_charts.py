"""Generate benchmark bar charts for README."""

import matplotlib.pyplot as plt
import numpy as np

# ── Light theme ─────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#ccc",
        "axes.labelcolor": "#333",
        "text.color": "#333",
        "xtick.color": "#555",
        "ytick.color": "#555",
        "grid.color": "#e0e0e0",
        "grid.alpha": 0.8,
        "font.family": "monospace",
        "font.size": 11,
    }
)

BM25S_COLOR = "#bbb"
BM25X_COLOR = "#2d8cf0"

# ── Benchmark data (BEIR datasets, measured on Apple M3) ────────────
datasets = [
    "NFCorpus\n3.6k docs",
    "SciFact\n5k docs",
    "SciDocs\n26k docs",
    "FiQA\n58k docs",
    "MS MARCO\n8.8M docs",
]

# NDCG@10
ndcg_bm25s = [0.3064, 0.6617, 0.1538, 0.2326, 0.2124]
ndcg_bm25x = [0.3287, 0.6904, 0.1600, 0.2514, 0.2186]

# Index throughput (docs/s)
index_tput_bm25s = [13_658, 15_138, 17_567, 23_698, 23_395]
index_tput_bm25x = [70_621, 77_644, 94_390, 144_060, 82_910]

# Search throughput (queries/s)
search_tput_bm25s = [36_992, 18_969, 6_543, 2_431, 16]
search_tput_bm25x = [128_245, 25_992, 7_032, 4_760, 65]


def _fmt_tput(v):
    if v >= 1_000:
        return f"{v / 1_000:,.0f}k"
    return f"{v:,.0f}"


def grouped_bar(
    ax,
    labels,
    vals_s,
    vals_x,
    ylabel,
    title,
    fmt="{:.4f}",
    show_legend=True,
    log_scale=False,
    abs_s=None,
    abs_x=None,
    abs_unit="",
):
    x = np.arange(len(labels))
    w = 0.35

    bars_s = ax.bar(
        x - w / 2,
        vals_s,
        w,
        label="bm25s",
        color=BM25S_COLOR,
        edgecolor="#ffffff",
        linewidth=0.5,
        zorder=3,
    )
    bars_x = ax.bar(
        x + w / 2,
        vals_x,
        w,
        label="bm25x",
        color=BM25X_COLOR,
        edgecolor="#ffffff",
        linewidth=0.5,
        zorder=3,
    )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", color="#111", pad=12)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Value labels on bars
    for i, (bar_s, bar_x, vs, vx) in enumerate(zip(bars_s, bars_x, vals_s, vals_x)):
        label_s = fmt.format(vs)
        label_x = fmt.format(vx)
        ax.text(
            bar_s.get_x() + bar_s.get_width() / 2,
            bar_s.get_height(),
            label_s,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#888",
        )
        ax.text(
            bar_x.get_x() + bar_x.get_width() / 2,
            bar_x.get_height(),
            label_x,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#2d8cf0",
            fontweight="bold",
        )

        # Absolute throughput inside bars
        if abs_s is not None and abs_x is not None:
            txt_s = f"{_fmt_tput(abs_s[i])} {abs_unit}"
            txt_x = f"{_fmt_tput(abs_x[i])} {abs_unit}"
            ax.text(
                bar_s.get_x() + bar_s.get_width() / 2,
                bar_s.get_height() / 2,
                txt_s,
                ha="center",
                va="center",
                fontsize=6.5,
                color="#555",
                rotation=90,
            )
            ax.text(
                bar_x.get_x() + bar_x.get_width() / 2,
                bar_x.get_height() / 2,
                txt_x,
                ha="center",
                va="center",
                fontsize=6.5,
                color="#fff",
                fontweight="bold",
                rotation=90,
            )

    if show_legend:
        ax.legend(
            loc="upper left",
            fontsize=9,
            facecolor="#fff",
            edgecolor="#ccc",
            labelcolor="#333",
        )

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Compute speedups (bm25s = 1x baseline) ─────────────────────────
index_speedup = [x / s for s, x in zip(index_tput_bm25s, index_tput_bm25x)]
search_speedup = [x / s for s, x in zip(search_tput_bm25s, search_tput_bm25x)]
baseline = [1.0] * len(datasets)

# ── Generate figure ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.subplots_adjust(wspace=0.3, left=0.05, right=0.97, top=0.88, bottom=0.15)

grouped_bar(
    axes[0],
    datasets,
    ndcg_bm25s,
    ndcg_bm25x,
    "NDCG@10",
    "Retrieval Quality (NDCG@10)",
    fmt="{:.4f}",
    show_legend=True,
)

grouped_bar(
    axes[1],
    datasets,
    baseline,
    index_speedup,
    "speedup vs bm25s",
    "Indexing Speedup",
    fmt="{:.1f}x",
    show_legend=False,
    abs_s=index_tput_bm25s,
    abs_x=index_tput_bm25x,
    abs_unit="d/s",
)

grouped_bar(
    axes[2],
    datasets,
    baseline,
    search_speedup,
    "speedup vs bm25s",
    "Search Speedup",
    fmt="{:.1f}x",
    show_legend=False,
    abs_s=search_tput_bm25s,
    abs_x=search_tput_bm25x,
    abs_unit="q/s",
)

fig.savefig("assets/benchmarks.png", dpi=200, facecolor="#ffffff")
print("Saved assets/benchmarks.png")
