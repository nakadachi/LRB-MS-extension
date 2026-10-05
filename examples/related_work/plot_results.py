"""Figure (logical error rate vs p) and markdown tables (error rate, time per shot) for the
related-work comparison. Reads results_<code>.json; writes docs/figures/related_work.png and
results_tables.md.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURE = HERE.parents[1] / "docs" / "figures" / "related_work.png"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# decoder: (colour, marker, line style); colours follow the fixed categorical order
STYLES = {
    "MBP4+LRB-MS-8 (ours)": ("#2a78d6", "o", "-"),
    "MBP4+LRB-MS-6 (ours)": ("#2a78d6", "o", "-"),
    "MBP4+LRB-MS-8+ladder (ours)": ("#2a78d6", "*", "--"),
    "GMBP4+OSD-1 [Mostad et al.]": ("#eb6834", "s", "-"),
    "SOGRAND+XZ [Rapp et al.]": ("#1baf7a", "D", "-"),
    "SOGRAND [Rapp et al.]": ("#1baf7a", "D", "--"),
    "SOGRAND+XZ+ladder [Rapp et al. + ours]": ("#1baf7a", "*", ":"),
    "LEAD [Xiao et al.]": ("#eda100", "^", "-"),
    "LEAD α=0.01 [Xiao et al.]": ("#eda100", "^", "--"),
    "MBP4+OSD-1": ("#e87ba4", "v", "--"),
    "BP+OSD-CS7": ("#52514e", "x", "--"),
    "MBP4+MAP, 100 it. (exact GC)": ("#4a3aa7", "P", ":"),
}
PANELS = [
    ("qt250", "[[250,10,15]] (Radebold et al.), 6 checks / vertex"),
    ("qt432", "[[432,16,28]] (Wang et al.), 12 checks / vertex"),
    ("qtC16", "[[576,32,≤16]] C16, 9 checks / vertex"),
]
LABELS = {
    "qt250": "[[250,10,15]]",
    "qt432": "[[432,16,28]]",
    "qtC16": "[[576,32,≤16]] C16",
    "bb144": "BB [[144,12,12]]",
}


def wilson_upper(failures, shots, z=1.96):
    rate = failures / shots
    denominator = 1 + z * z / shots
    centre = (rate + z * z / (2 * shots)) / denominator
    half_width = z * np.sqrt(rate * (1 - rate) / shots + z * z / (4 * shots**2)) / denominator
    return centre + half_width


def load(code):
    path = HERE / f"results_{code}.json"
    return json.load(open(path)) if path.exists() else []


def plot():
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
    fig, axes = plt.subplots(1, len(PANELS), figsize=(12, 3.9), facecolor=SURFACE)
    handles = {}
    for ax, (code, title) in zip(axes, PANELS):
        results = load(code)
        for decoder in dict.fromkeys(r["decoder"] for r in results):
            color, marker, style = STYLES.get(decoder, ("#888888", ".", "-"))
            points = sorted((r["p"], r) for r in results if r["decoder"] == decoder)
            ps = np.array([p for p, _ in points])
            fails = np.array([r["fails"] for _, r in points])
            shots = np.array([r["shots"] for _, r in points])
            zero = fails == 0
            (line,) = ax.plot(
                ps[~zero],
                fails[~zero] / shots[~zero],
                style,
                color=color,
                marker=marker,
                ms=5,
                lw=2,
                label=decoder,
            )
            ax.plot(
                ps[zero],
                [wilson_upper(0, s) for s in shots[zero]],
                "v",
                color=color,
                ms=6,
                mfc="none",
                mew=1.4,
            )
            handles.setdefault(decoder, line)
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.set_xlabel("depolarizing error rate p")
    axes[0].set_ylabel("logical error rate per shot")
    fig.legend(
        handles.values(),
        handles.keys(),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        handlelength=3.2,
    )
    fig.tight_layout()
    fig.text(
        0.5,
        -0.02,
        "Same depolarizing errors for every decoder. Open triangles: no failures, 95% upper bound.",
        ha="center",
        va="top",
        color=INK2,
        fontsize=8,
    )
    fig.savefig(FIGURE, dpi=150, facecolor=SURFACE, bbox_inches="tight")


def format_rate(result):
    if result["fails"] == 0:
        return f"0 (/{result['shots']})"
    return f"{result['fails'] / result['shots']:.1e}"


def tables():
    lines = []
    for code in ("qt250", "qt432", "qtC16", "bb144"):
        results = load(code)
        if not results:
            continue
        ps = sorted({r["p"] for r in results})
        decoders = list(dict.fromkeys(r["decoder"] for r in results))
        for metric, title in (("ler", "logical error rate"), ("time", "mean decode time per shot")):
            lines.append(f"\n**{LABELS[code]}**, {title}\n")
            lines.append("| decoder | " + " | ".join(f"p = {p:g}" for p in ps) + " |")
            lines.append("|---" * (len(ps) + 1) + "|")
            for decoder in decoders:
                cells = []
                for p in ps:
                    match = [r for r in results if r["decoder"] == decoder and r["p"] == p]
                    if not match:
                        cells.append("—")
                    elif metric == "ler":
                        cells.append(format_rate(match[0]))
                    else:
                        cells.append(f"{match[0]['ms_per_shot']:.2f} ms")
                lines.append(f"| {decoder} | " + " | ".join(cells) + " |")
    (HERE / "results_tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    plot()
    tables()
