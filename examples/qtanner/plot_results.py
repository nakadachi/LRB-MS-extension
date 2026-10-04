"""Plot results.json (logical error rate and decode time vs p) and write a markdown table."""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# categorical colour slots in fixed order
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# greedy overlap grouping with ell=9 recovers exactly the vertex groups on these codes,
# so its curves coincide with "LRB-MS-8 (vertex GCs)"; it stays in the table only
PLOT_EXCLUDE = {"LRB-MS-8 (greedy ell=9)"}


def wilson(failures, shots, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if shots == 0:
        return 0.0, 0.0
    rate = failures / shots
    denominator = 1 + z * z / shots
    centre = (rate + z * z / (2 * shots)) / denominator
    half_width = z * np.sqrt(rate * (1 - rate) / shots + z * z / (4 * shots * shots)) / denominator
    return max(centre - half_width, 0.0), centre + half_width


def plot_panel(ax, results, code, decoder, metric, color, style, with_label):
    points = sorted((r["p"], r) for r in results if r["code"] == code and r["decoder"] == decoder)
    if not points:
        return
    ps = np.array([p for p, _ in points])
    if metric == "time":
        times_ms = [r["us_per_shot"] / 1000 for _, r in points]
        ax.plot(ps, times_ms, style, color=color, lw=2, marker="o", ms=4)
        return
    ler = np.array([r["fails"] / r["shots"] for _, r in points])
    low, high = np.array([wilson(r["fails"], r["shots"]) for _, r in points]).T
    # zero-failure points: plot the 95% upper bound as an open marker
    zero = ler == 0
    ax.plot(
        ps[~zero],
        ler[~zero],
        style,
        color=color,
        lw=2,
        marker="o",
        ms=4,
        label=decoder if with_label else None,
    )
    ax.fill_between(ps[~zero], low[~zero], high[~zero], color=color, alpha=0.12, lw=0)
    ax.plot(ps[zero], high[zero], "v", color=color, ms=5, mfc="none", mew=1.2)


def write_table(results, codes, decoders, table_path):
    lines = []
    for code in codes:
        ps = sorted({r["p"] for r in results if r["code"] == code})
        lines.append(f"\n**{code}**: logical error rate (failures/shots), mean decode time\n")
        lines.append("| decoder | " + " | ".join(f"p={p}" for p in ps) + " |")
        lines.append("|---" * (len(ps) + 1) + "|")
        for decoder in decoders:
            cells = []
            for p in ps:
                match = [
                    r
                    for r in results
                    if r["code"] == code and r["decoder"] == decoder and r["p"] == p
                ]
                if not match:
                    cells.append("—")
                    continue
                r = match[0]
                cells.append(
                    f"{r['fails'] / r['shots']:.1e} ({r['fails']}/{r['shots']}), "
                    f"{r['us_per_shot'] / 1000:.2f} ms"
                )
            lines.append(f"| {decoder} | " + " | ".join(cells) + " |")
    open(table_path, "w").write("\n".join(lines) + "\n")


def main(
    path="results.json",
    codes_path="codes.json",
    out="qtanner_benchmark.png",
    title="Quantum Tanner codes",
    table="results_table.md",
    noise="code-capacity X noise, decoded with H_Z",
    xlabel="physical bit-flip rate p",
):
    results = json.load(open(path))
    codes = [entry["label"] for entry in json.load(open(codes_path))]
    table_decoders = list(dict.fromkeys(r["decoder"] for r in results))
    decoders = [d for d in table_decoders if d not in PLOT_EXCLUDE]
    colors = {d: SLOTS[i % len(SLOTS)] for i, d in enumerate(decoders)}
    styles = {d: ("--" if d.startswith("BP") else "-") for d in decoders}

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
        }
    )
    fig, axes = plt.subplots(
        2,
        len(codes),
        figsize=(max(4.2 * len(codes), 7.0), 7.2),
        sharex=True,
        squeeze=False,
        facecolor=SURFACE,
        gridspec_kw=dict(height_ratios=[3, 2]),
    )
    for column, code in enumerate(codes):
        for row, metric in enumerate(["ler", "time"]):
            ax = axes[row, column]
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for decoder in decoders:
                plot_panel(
                    ax,
                    results,
                    code,
                    decoder,
                    metric,
                    colors[decoder],
                    styles[decoder],
                    with_label=column == 0,
                )
            ax.set_yscale("log")
            if row == 0:
                ax.set_title(code, fontsize=10, color=INK)
                ax.set_ylim(1e-5, 1.2)
            else:
                ax.set_xlabel(xlabel)
    axes[0, 0].set_ylabel("logical error rate (per shot)")
    axes[1, 0].set_ylabel("decode time (ms / shot, 1 core)")
    narrow = len(codes) == 1
    fig.legend(
        loc="lower center", ncol=2 if narrow else 4, frameon=False, bbox_to_anchor=(0.5, -0.01)
    )
    fig.text(
        0.5,
        0.995,
        f"{title}, {noise}"
        + ("\n" if narrow else "  ")
        + "(open triangles: 0 failures, 95% upper bound)",
        ha="center",
        va="top",
        color=INK2,
    )
    fig.tight_layout(rect=(0, 0.13 if narrow else 0.07, 1, 0.95 if narrow else 0.97))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    write_table(results, codes, table_decoders, table)


if __name__ == "__main__":
    main(*sys.argv[1:])
