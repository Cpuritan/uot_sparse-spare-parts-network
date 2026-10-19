"""Publication figures from saved observations; SAA uses continuous-inventory records."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from environment import OUT

POLICIES = ["SD-TV-UOT", "MVP-C", "MVP-I", "SAA-C"]
LABELS = {method: "SAA" if method == "SAA-C" else method for method in POLICIES}
COLORS = ["#0F4D92", "#5B9F68", "#9BC99A", "#B64342"]
TITLES = {"S": "Small scale", "M": "Medium scale", "L": "Large scale"}


def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 13, "axes.linewidth": 1.1,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.facecolor": "white"})


def save(figure, name):
    for extension in ["png", "pdf", "svg"]:
        figure.savefig(OUT / f"{name}.{extension}", dpi=300, bbox_inches="tight", pad_inches=.08)
    plt.close(figure)


def make_figures(summary, observations):
    style()
    summary = summary[summary.method.isin(POLICIES)].copy()
    observations = observations[observations.method.isin(POLICIES)].copy()
    summary.assign(method=summary.method.map(LABELS)).to_csv(
        OUT / "figure-exp2-2-paper-data.csv", index=False, encoding="utf-8-sig")
    observations.assign(method=observations.method.map(LABELS)).to_csv(
        OUT / "figure-exp2-1-paper-data.csv", index=False, encoding="utf-8-sig")
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.5))
    for panel, (scale, axis) in enumerate(zip("SML", axes)):
        for position, (method, color) in enumerate(zip(POLICIES, COLORS), 1):
            values = observations.loc[(observations.scale == scale) & (observations.method == method), "total_cost"].to_numpy()
            if not len(values):
                axis.text(position, .04, "Unavailable", transform=axis.get_xaxis_transform(), rotation=90,
                          ha="center", va="bottom", fontsize=9)
                continue
            axis.boxplot([values], positions=[position], widths=.55, patch_artist=True, showmeans=True,
                boxprops={"facecolor": color, "edgecolor": color, "alpha": .55, "linewidth": 1.3},
                medianprops={"color": "#272727", "linewidth": 1.4},
                whiskerprops={"color": color, "linewidth": 1.2}, capprops={"color": color, "linewidth": 1.2},
                meanprops={"marker": "D", "markerfacecolor": color, "markeredgecolor": "#272727", "markersize": 4.5},
                flierprops={"marker": "o", "markerfacecolor": "none", "markeredgecolor": color,
                            "markersize": 2.6, "markeredgewidth": .6, "alpha": .65})
        axis.set_title(f"({chr(97 + panel)}) {TITLES[scale]}", loc="left", pad=12)
        axis.set_xticks(range(1, len(POLICIES) + 1), [LABELS[method] for method in POLICIES],
                        rotation=28, ha="right", fontsize=10.5)
        axis.set_xlim(.4, len(POLICIES) + .6)
        axis.set_xlabel("Policy")
        axis.set_ylabel("Out-of-sample total cost (€)")
        axis.grid(axis="y", color="#e3e3e3", linewidth=.55)
        axis.set_axisbelow(True)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    figure.legend(handles=[Line2D([], [], marker="D", color="none", markerfacecolor="#767676",
                   markeredgecolor="#272727", markersize=5, label="Mean"),
                   Line2D([], [], color="#272727", linewidth=1.4, label="Median")],
                   loc="lower center", ncol=2, bbox_to_anchor=(.5, -.005), fontsize=10)
    figure.tight_layout(pad=1.5, rect=(0, .065, 1, 1))
    save(figure, "figure-exp2-1")
    figure, axis = plt.subplots(figsize=(8.6, 4.8))
    for method, color, marker, linestyle in zip(POLICIES, COLORS, ["o", "s", "^", "D"], ["-", "-", "--", "-"]):
        times = [float(summary.loc[(summary.scale == scale) & (summary.method == method), "planning_seconds"].iloc[0])
                 for scale in "SML"]
        axis.plot(range(3), times, marker=marker, markersize=6.5, linewidth=1.9,
                  linestyle=linestyle, color=color, label=LABELS[method], markeredgecolor="#4D4D4D", markeredgewidth=.5)
    axis.set_yscale("log")
    axis.set_ylabel("Solving time (s)")
    axis.set_xlabel("Instance scale")
    axis.set_xticks(range(3), list("SML"))
    axis.set_xlim(-.15, 2.15)
    axis.grid(axis="y", which="major", color="#e3e3e3", linewidth=.65)
    axis.legend(loc="upper center", bbox_to_anchor=(.5, 1.16), ncol=len(POLICIES), fontsize=10.5,
                columnspacing=1.35, handlelength=2., handletextpad=.5)
    figure.tight_layout(pad=1.5)
    save(figure, "figure-exp2-2")


if __name__ == "__main__":
    import pandas as pd
    make_figures(pd.read_csv(OUT / "table-exp2-summary.csv"),
                 pd.read_csv(OUT / "figure-exp2-1-data.csv"))
