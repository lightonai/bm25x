"""Generate benchmark bar charts for README — bm25s vs bm25x (CPU/GPU/batch)."""

import matplotlib.pyplot as plt
import numpy as np

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

COLORS = {
    "bm25s": "#bbb",
    "bm25x": "#2d8cf0",
    "bm25x GPU": "#22c55e",
    "bm25x batch": "#93c5fd",
    "4xGPU batch": "#15803d",
}

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
ndcg_bm25x_gpu = [0.3287, 0.6904, 0.1600, 0.2514, 0.2240]

# Index throughput (docs/s)
index_tput_bm25s = [13_658, 15_138, 17_567, 23_698, 23_395]
index_tput_bm25x = [70_621, 77_644, 94_390, 144_060, 82_910]
index_tput_bm25x_gpu = [32_551, 57_570, 68_360, 205_115, 295_458]

# Search throughput (queries/s)
search_bm25s = [36_992, 18_969, 6_543, 2_431, 16]
search_bm25x = [128_245, 25_992, 7_032, 4_760, 65]
search_bm25x_gpu = [6_940, 5_682, 6_197, 5_935, 3_430]
search_bm25x_batch = [168_733, 72_805, 24_541, 15_624, 96]
search_bm25x_4gpu = [20_089, 17_074, 18_780, 17_336, 13_047]


def _fmt_tput(v):
    if v >= 1_000:
        return f"{v / 1_000:,.0f}k"
    return f"{v:,.0f}"


def grouped_bar(
    ax,
    labels,
    series,
    ylabel,
    title,
    fmt="{:.4f}",
    log_scale=False,
    abs_values=None,
    abs_unit="",
    bar_width_scale=1.0,
):
    n_groups = len(labels)
    n_series = len(series)
    w = 0.8 / n_series * bar_width_scale
    x = np.arange(n_groups)

    bars_all = []
    for i, (name, vals) in enumerate(series):
        offset = (i - n_series / 2 + 0.5) * w
        color = COLORS.get(name, "#999")
        bars = ax.bar(
            x + offset,
            vals,
            w,
            label=name,
            color=color,
            edgecolor="#ffffff",
            linewidth=0.5,
            zorder=3,
        )
        bars_all.append((name, bars, vals))

    if log_scale:
        ax.set_yscale("log")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", color="#111", pad=12)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    for name, bars, vals in bars_all:
        color = COLORS.get(name, "#999")
        text_color = color if color != "#bbb" else "#888"
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=6,
                color=text_color,
                fontweight="bold",
            )

    if abs_values:
        for (name, bars, _), abs_v in zip(bars_all, abs_values):
            color_inside = "#555" if name == "bm25s" else "#fff"
            for bar, av in zip(bars, abs_v):
                h = bar.get_height()
                if h > (0.3 if not log_scale else 0):
                    y = h**0.5 if log_scale else h / 2
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        y,
                        f"{_fmt_tput(av)} {abs_unit}",
                        ha="center",
                        va="center",
                        fontsize=5,
                        color=color_inside,
                        fontweight="bold",
                        rotation=90,
                    )

    ax.legend(loc="upper left", fontsize=10, facecolor="#fff", edgecolor="#ccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


baseline = [1.0] * 5
idx_su_x = [x / s for s, x in zip(index_tput_bm25s, index_tput_bm25x)]
idx_su_gpu = [x / s for s, x in zip(index_tput_bm25s, index_tput_bm25x_gpu)]
s_su_x = [x / s for s, x in zip(search_bm25s, search_bm25x)]
s_su_gpu = [x / s for s, x in zip(search_bm25s, search_bm25x_gpu)]
s_su_batch = [x / s for s, x in zip(search_bm25s, search_bm25x_batch)]
s_su_4gpu = [x / s for s, x in zip(search_bm25s, search_bm25x_4gpu)]

# ── Layout: 2 big charts on top, 1 smaller chart below ────────────
fig, (ax_index, ax_search, ax_ndcg) = plt.subplots(3, 1, figsize=(16, 22))
fig.subplots_adjust(hspace=0.35, left=0.08, right=0.95, top=0.96, bottom=0.04)

# Indexing speedup (top left)
grouped_bar(
    ax_index,
    datasets,
    [("bm25s", index_tput_bm25s), ("bm25x", index_tput_bm25x), ("bm25x GPU", index_tput_bm25x_gpu)],
    "throughput — docs/s (log)",
    "Indexing Speed (d/s)",
    fmt="{:,.0f}",
    log_scale=True,
)

# Search throughput
grouped_bar(
    ax_search,
    datasets,
    [
        ("bm25s", search_bm25s),
        ("bm25x", search_bm25x),
        ("bm25x batch", search_bm25x_batch),
        ("bm25x GPU", search_bm25x_gpu),
        ("4xGPU batch", search_bm25x_4gpu),
    ],
    "throughput — queries/s (log)",
    "Search Speed (q/s)",
    fmt="{:,.0f}",
    log_scale=True,
)

# Retrieval quality (bottom, full width)
grouped_bar(
    ax_ndcg,
    datasets,
    [("bm25s", ndcg_bm25s), ("bm25x", ndcg_bm25x), ("bm25x GPU", ndcg_bm25x_gpu)],
    "NDCG@10",
    "Retrieval Quality (NDCG@10)",
    fmt="{:.4f}",
    bar_width_scale=0.55,
)

fig.savefig("assets/benchmarks.png", dpi=200, facecolor="#ffffff")
print("Saved assets/benchmarks.png")
