"""Plot results.json (logical error rate and decode time vs p) and write a markdown table."""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# categorical slots in fixed order
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    ph = k / n
    den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(mid - half, 0.0), mid + half


# greedy overlap grouping with ell=9 recovers exactly the vertex groups on these codes,
# so its curves coincide with "LRB-MS-8 (vertex GCs)"; it stays in the table only
PLOT_EXCLUDE = {"LRB-MS-8 (greedy ell=9)"}


def main(path="results.json", codes_path="codes.json", out="qtanner_benchmark.png",
         title="Quantum Tanner codes", table="results_table.md",
         noise="code-capacity X noise, decoded with H_Z", xlabel="physical bit-flip rate p"):
    res = json.load(open(path))
    codes = [c["label"] for c in json.load(open(codes_path))]
    decoders = list(dict.fromkeys(r["decoder"] for r in res))
    table_decoders = decoders
    decoders = [d for d in decoders if d not in PLOT_EXCLUDE]
    color = {d: SLOTS[i % len(SLOTS)] for i, d in enumerate(decoders)}
    style = {d: ("--" if d.startswith("BP") else "-") for d in decoders}

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, axes = plt.subplots(2, len(codes), figsize=(max(4.2 * len(codes), 7.0), 7.2), sharex=True, squeeze=False,
                             facecolor=SURFACE, gridspec_kw=dict(height_ratios=[3, 2]))
    for j, code in enumerate(codes):
        for row, metric in enumerate(["ler", "time"]):
            ax = axes[row, j]
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for d in decoders:
                pts = sorted((r["p"], r) for r in res if r["code"] == code and r["decoder"] == d)
                if not pts:
                    continue
                ps = np.array([p for p, _ in pts])
                if metric == "ler":
                    ler = np.array([r["fails"] / r["shots"] for _, r in pts])
                    lo, hi = np.array([wilson(r["fails"], r["shots"]) for _, r in pts]).T
                    # zero-failure points: plot the 95% upper bound as an open marker
                    zero = ler == 0
                    ax.plot(ps[~zero], ler[~zero], style[d], color=color[d], lw=2, marker="o", ms=4,
                            label=d if j == 0 else None)
                    ax.fill_between(ps[~zero], lo[~zero], hi[~zero], color=color[d], alpha=0.12, lw=0)
                    ax.plot(ps[zero], hi[zero], "v", color=color[d], ms=5, mfc="none", mew=1.2)
                else:
                    ax.plot(ps, [r["us_per_shot"] / 1000 for _, r in pts], style[d], color=color[d],
                            lw=2, marker="o", ms=4)
            ax.set_yscale("log")
            if row == 0:
                ax.set_title(code, fontsize=10, color=INK)
                ax.set_ylim(1e-5, 1.2)
            else:
                ax.set_xlabel(xlabel)
        axes[0, 0].set_ylabel("logical error rate (per shot)")
        axes[1, 0].set_ylabel("decode time (ms / shot, 1 core)")
    narrow = len(codes) == 1
    fig.legend(loc="lower center", ncol=2 if narrow else 4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, 0.995, f"{title}, {noise}" + ("\n" if narrow else "  ") +
             "(open triangles: 0 failures, 95% upper bound)", ha="center", va="top", color=INK2)
    fig.tight_layout(rect=(0, 0.13 if narrow else 0.07, 1, 0.95 if narrow else 0.97))
    fig.savefig(out, dpi=150, facecolor=SURFACE)

    lines = []
    for code in codes:
        ps = sorted({r["p"] for r in res if r["code"] == code})
        lines.append(f"\n**{code}**: logical error rate (failures/shots), mean decode time\n")
        lines.append("| decoder | " + " | ".join(f"p={p}" for p in ps) + " |")
        lines.append("|---" * (len(ps) + 1) + "|")
        for d in table_decoders:
            cells = []
            for p in ps:
                r = next((r for r in res if r["code"] == code and r["decoder"] == d and r["p"] == p), None)
                cells.append("—" if r is None else
                             f"{r['fails'] / r['shots']:.1e} ({r['fails']}/{r['shots']}), "
                             f"{r['us_per_shot'] / 1000:.2f} ms")
            lines.append(f"| {d} | " + " | ".join(cells) + " |")
    open(table, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
