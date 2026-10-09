"""Headline figures for the README, built from the benchmark result files.

Usage (from the repository root): python docs/figures/make_figures.py
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def wilson_upper(failures, shots, z=1.96):
    """Upper end of the Wilson score interval (used for zero-failure points)."""
    rate = failures / shots
    denominator = 1 + z * z / shots
    centre = (rate + z * z / (2 * shots)) / denominator
    half_width = z * np.sqrt(rate * (1 - rate) / shots + z * z / (4 * shots**2)) / denominator
    return centre + half_width


def style_axis(ax, title, xlabel):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel(xlabel)


def plot_ler(ax, results, code, decoder, color, marker, label):
    """Logical error rate vs p; zero-failure points drawn as open triangles at the 95% bound."""
    points = sorted((r["p"], r) for r in results if r["code"] == code and r["decoder"] == decoder)
    if not points:
        return
    ps = np.array([p for p, _ in points])
    fails = np.array([r["fails"] for _, r in points])
    shots = np.array([r["shots"] for _, r in points])
    zero = fails == 0
    ax.plot(
        ps[~zero], fails[~zero] / shots[~zero], color=color, lw=2, marker=marker, ms=5, label=label
    )
    bounds = [wilson_upper(0, s) for s in shots[zero]]
    ax.plot(ps[zero], bounds, "v", color=color, ms=6, mfc="none", mew=1.4)


def top_legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )


def save(fig, name, note):
    fig.tight_layout()
    fig.text(0.5, -0.02, note, ha="center", va="top", color=INK2, fontsize=8)
    fig.savefig(OUT / name, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def figure_large_tanner():
    results = json.load(open(ROOT / "examples/qtanner/results_large.json"))
    codes = [
        ("A5/[8,4,4] [[3840,48]]", "[[3840,48]], [8,4,4] local code"),
        ("PSL(2,7)/[6,3,3] [[6048,40]]", "[[6048,40]], [6,3,3] local code"),
        ("PSL(2,7)/[8,4,4] [[10752,56]]", "[[10752,56]], [8,4,4] local code"),
    ]
    series = [
        ("Qulid-8 (vertex GCs)", BLUE, "o", "Qulid-8"),
        ("BP (min-sum)", ORANGE, "s", "BP (min-sum)"),
        ("BP+LSD-CS7", AQUA, "D", "BP+LSD-CS7"),
        ("BP+OSD-CS7", YELLOW, "^", "BP+OSD-CS7"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True, facecolor=SURFACE)
    for ax, (code, title) in zip(axes, codes):
        for decoder, color, marker, label in series:
            plot_ler(ax, results, code, decoder, color, marker, label)
        style_axis(ax, title, "bit-flip rate p")
        ax.set_ylim(5e-5, 1.5)
    axes[0].set_ylabel("logical error rate per shot")
    top_legend(fig, axes[0])
    save(
        fig,
        "tanner_large.png",
        "Code-capacity X noise decoded with H_Z. Open triangles: no failures, 95% upper bound.",
    )


def figure_bb_ensemble():
    capacity = json.load(open(ROOT / "examples/bb/results_large.json"))
    circuit = json.load(open(ROOT / "examples/bb_circuit/results.json"))
    panels = [
        (capacity, "[[360,12,<=24]]", "[[360,12,≤24]], code capacity", "bit-flip rate p"),
        (capacity, "[[756,16,<=34]]", "[[756,16,≤34]], code capacity", "bit-flip rate p"),
        (
            circuit,
            "[[144,12,12]] x12 rounds",
            "[[144,12,12]], circuit level, 12 rounds",
            "circuit error rate p",
        ),
    ]
    series = [
        ("Ensemble x8, all, OSD on all", BLUE, "o", "Qulid ensemble ×8, all"),
        ("Ensemble x8, first, OSD on last", VIOLET, "s", "Qulid ensemble ×8, first"),
        ("BP+OSD-CS40", YELLOW, "^", "BP+OSD-CS40"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), facecolor=SURFACE)
    for ax, (results, code, title, xlabel) in zip(axes, panels):
        for decoder, color, marker, label in series:
            plot_ler(ax, results, code, decoder, color, marker, label)
        style_axis(ax, title, xlabel)
    axes[2].set_ylim(1e-3, 1e-1)
    axes[0].set_ylabel("logical error rate per shot")
    top_legend(fig, axes[0])
    save(fig, "bb_ensemble.png", "Open triangles: no failures, 95% upper bound.")


def figure_relay_ladder():
    # [[288,12,18]], total depolarizing p, MBP4 + Qulid (README: relay ladders)
    rungs = ["base (0.75, 1)", "+ μ-only retries", "+ (μ, α) retries"]
    runs = [
        ("p = 0.075, 120k shots", [97 / 120e3, 21 / 120e3, 9 / 120e3], ORANGE, "s"),
        ("p = 0.065, 600k shots", [107 / 600e3, 18 / 600e3, 9 / 600e3], BLUE, "o"),
        ("p = 0.06, 600k shots, LRB order 1", [20 / 600e3, 3 / 600e3, 1 / 600e3], VIOLET, "D"),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.6), facecolor=SURFACE)
    x = np.arange(len(rungs))
    for label, rates, color, marker in runs:
        ax.plot(x, rates, color=color, lw=2, marker=marker, ms=6, label=label)
        ax.annotate(
            f"×{rates[0] / rates[-1]:.0f}",
            (x[-1], rates[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=INK2,
        )
    style_axis(ax, "[[288,12,18]], depolarizing noise, MBP4 + Qulid", "")
    ax.set_xticks(x, rungs)
    ax.set_xlim(-0.2, len(rungs) - 0.6)
    ax.set_ylabel("logical error rate per shot")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "relay_ladder.png", "Retries fire only when the previous attempt did not converge.")


if __name__ == "__main__":
    figure_large_tanner()
    figure_bb_ensemble()
    figure_relay_ladder()
