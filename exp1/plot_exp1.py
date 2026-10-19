"""Two publication figures, drawn only from this experiment's saved results."""
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from environment import OUT, SETTINGS, instance, write_json
from geometry import nominal_regions

PALETTE = ["#0F4D92", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#332288", "#882255"]


def style():
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11, "axes.linewidth": 1.1,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": "white", "axes.unicode_minus": True})


def export(figure, name):
    for extension in ["png", "pdf", "svg"]:
        figure.savefig(OUT / f"{name}.{extension}", dpi=300, bbox_inches="tight", pad_inches=.12)
    plt.close(figure)


def evolution(record):
    inst = instance("M", SETTINGS["fixed_seed"])
    history = np.asarray(record["inventory_history"])
    final = np.asarray(record["inventory"])
    thresholds = np.asarray(record["threshold_history"])
    terminal = record["iterations"]
    active = np.asarray(record["active"], int)
    colors = {warehouse: PALETTE[position % len(PALETTE)] for position, warehouse in enumerate(active)}
    map_colors = {warehouse: colors.get(warehouse, plt.get_cmap("tab20")(warehouse % 20))
                  for warehouse in range(len(inst.sites))}
    figure, axes = plt.subplots(2, 4, figsize=(16.8, 8.5))
    figure.subplots_adjust(left=.055, right=.99, top=.94, bottom=.10, wspace=.32, hspace=.52)
    ticks = sorted(set([0, 5, 10, 15, terminal]))
    ticks = [tick for tick in ticks if tick <= terminal - 2 or tick == terminal]
    active_counts = np.count_nonzero(history.sum(axis=2) > 0, axis=1)
    count_axis = axes[0, 0]
    count_axis.step(np.arange(terminal + 1), active_counts, where="post", color=PALETTE[0], lw=2)
    count_axis.plot([terminal, terminal + 1], [active_counts[-1], len(active)], "--", color=PALETTE[0])
    count_axis.scatter([terminal + 1], [len(active)], marker="s", color=PALETTE[0], zorder=5)
    count_axis.set(title="(a) Warehouse activation", ylabel="Number of active warehouses")
    count_axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    count_axis.set_ylim(0, len(inst.sites) + 1)
    limit = max(history.max(), final.max()) * 1.12
    for part, axis in enumerate(axes[0, 1:3]):
        for warehouse in range(len(inst.sites)):
            color = colors.get(warehouse, "#C8C8C8")
            axis.plot(np.arange(terminal + 1), history[:, warehouse, part], color=color,
                      lw=1.8 if warehouse in colors else .85, zorder=3 if warehouse in colors else 1)
            axis.plot([terminal, terminal + 1], [history[-1, warehouse, part], final[warehouse, part]],
                      color=color, lw=1.2, ls="--")
            if warehouse in colors:
                axis.scatter([terminal + 1], [final[warehouse, part]], color=color, marker="s", s=23, zorder=4)
        axis.set(title=f"({chr(98 + part)}) Part {part + 1} inventory", ylabel="Inventory", ylim=(0, limit))
        axis.text(.04, .94, f"$Q_{part + 1}$ = {inst.part_q[part]:.2f}", transform=axis.transAxes, va="top", fontsize=10)
    for warehouse in range(len(inst.sites)):
        axes[0, 3].plot(np.arange(1, terminal + 1), thresholds[:, warehouse],
                        color=colors.get(warehouse, "#C8C8C8"), lw=1.8 if warehouse in colors else .85)
    axes[0, 3].set(title="(d) Shared design weights", ylabel=r"$w_i^{(t)}=-F_i/(s_i^{(t-1)}+\varepsilon)$")
    for axis in axes[0]:
        axis.set_xlabel("Iteration")
        axis.grid(axis="y", color="#ECECEC", lw=.7)
        axis.set_axisbelow(True)
        axis.set_xlim(-.5, terminal + 1.8)
        regular_ticks = [tick for tick in range(0, terminal + 1, 5) if tick <= terminal - 1]
        axis.set_xticks(regular_ticks + [terminal + 1], [str(tick) for tick in regular_ticks] + ["R"])
    axes[0, 3].set_xticks(ticks, [str(tick) for tick in ticks])
    maps = []
    offsets = {1: (7, 0), 2: (-6, 6), 4: (-5, 5), 5: (4, -6), 12: (-4, -10), 13: (4, 6), 15: (-7, 7), 16: (10, 5)}
    for position, (axis, step) in enumerate(zip(axes[1], [1, 3, 5, "R"])):
        if step == "R":
            regions, stock = record["regions"], final
        else:
            regions = nominal_regions(inst, np.ones(len(inst.sites), bool), -thresholds[step - 1])
            stock = history[step]
        maps.append({"step": step, "regions": regions})
        fills = [(.91, .91, .91) if region["option"] == 0 else
                 tuple(.32 * np.asarray(to_rgb(map_colors[region["option"] - 1])) + .68) for region in regions]
        axis.add_collection(PolyCollection([region["vertices"] for region in regions], facecolors=fills,
                                           edgecolors="none", linewidths=0, antialiased=False, rasterized=False))
        axis.scatter(inst.xy[:, 0], inst.xy[:, 1], s=9 + 22 * inst.probabilities.sum(axis=1),
                     color="#444444", alpha=.55, linewidths=0, zorder=3)
        axis.scatter(inst.sites[:, 0], inst.sites[:, 1], marker="x", s=33, color="#777777", linewidths=1, zorder=4)
        current = np.flatnonzero(stock.sum(axis=1) > 0)
        for warehouse in current:
            axis.scatter(*inst.sites[warehouse], marker="s", s=44, color=map_colors[warehouse],
                         edgecolors="white", linewidths=.7, zorder=5)
            axis.annotate(f"W{warehouse + 1}", inst.sites[warehouse], xytext=offsets.get(warehouse, (4, 5)),
                          textcoords="offset points", fontsize=8, color="#202020", zorder=6)
        title = "Recovery" if step == "R" else f"$t={step}$"
        axis.set(title=f"({chr(101 + position)}) {title}", xlabel="position (km)", ylabel="position (km)",
                 xlim=(0, inst.domain_size), ylim=(0, inst.domain_size), aspect="equal")
        axis.set_xticks([0, 15, 30, 45])
        axis.set_yticks([0, 15, 30, 45])
    warehouse_legend = [Line2D([], [], color=color, lw=2, label=f"W{warehouse + 1}") for warehouse, color in colors.items()]
    warehouse_legend.append(Line2D([], [], color="#C8C8C8", lw=2, label="Other candidates"))
    figure.legend(handles=warehouse_legend, loc="center", bbox_to_anchor=(.5, .515), ncol=len(warehouse_legend), fontsize=10)
    figure.legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor="#555555", markersize=4, label="Equipment"),
        Line2D([], [], marker="x", color="#777777", linestyle="none", label="Candidate warehouse"),
        Line2D([], [], marker="s", color="none", markerfacecolor=PALETTE[0], label="Active warehouse"),
        Patch(facecolor="#E8E8E8", label="External supply")], loc="lower center", bbox_to_anchor=(.5, .012), ncol=4, fontsize=10)
    write_json(OUT / "figure-exp1-1-data.json", {"scale": inst.scale, "seed": inst.seed,
        "inventory_history": history, "recovered_inventory": final, "threshold_history": thresholds,
        "active_counts": active_counts, "map_steps": [1, 3, 5, "R"], "maps": maps})
    export(figure, "figure-exp1-1")
    return {"panels": 8, "scale": "M", "seed": inst.seed, "map_steps": [1, 3, 5, "R"],
            "iterations": terminal, "active_count": len(active), "part_demand": inst.part_q}


def mc_deviation(data):
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    figure.subplots_adjust(left=.12, right=.98, bottom=.15, top=.90)
    summary = []
    for scale_position, scale in enumerate("SML"):
        for budget_position, budget in enumerate(SETTINGS["mc_sizes"]):
            group = data[(data.scale == scale) & (data.budget == budget)].sort_values("seed")
            assert group.seed.tolist() == SETTINGS["seeds"]
            assert (group.n_repeats == SETTINGS["trajectory_repeats"]).all()
            values = group.D.to_numpy() * 100
            center = scale_position + (budget_position - 1) * .24
            color = PALETTE[budget_position]
            axis.boxplot([values], positions=[center], widths=.18, patch_artist=True, showfliers=False,
                         manage_ticks=False, boxprops={"facecolor": color + "25", "edgecolor": color, "linewidth": 1.3},
                         medianprops={"color": color, "linewidth": 2}, whiskerprops={"color": color},
                         capprops={"color": color})
            jitter = np.random.default_rng(20261018 + scale_position * 3 + budget_position).uniform(-.065, .065, len(group))
            scatter = axis.scatter(center + jitter, values, s=20, color=color, alpha=.7, linewidths=.3,
                                   edgecolors="white", zorder=3)
            scatter.set_gid(f"mc_points_{scale}_{budget}_n20")
            summary.append({"scale": scale, "budget": budget, "instances": len(group),
                "D_min_percent": values.min(), "D_q1_percent": np.quantile(values, .25),
                "D_median_percent": np.median(values), "D_q3_percent": np.quantile(values, .75),
                "D_max_percent": values.max()})
    axis.set(xlim=(-.55, 2.55), ylim=(-1., data.D.max() * 110),
             xlabel="Instance scale", ylabel="Normalized trajectory deviation\n(% of total demand)")
    axis.set_xticks(range(3), list("SML"))
    axis.grid(axis="y", color="#E9E9E9", lw=.7)
    axis.set_axisbelow(True)
    handles = [Patch(facecolor=PALETTE[position] + "25", edgecolor=PALETTE[position], label=f"MC {budget:,}")
               for position, budget in enumerate(SETTINGS["mc_sizes"])]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, .995), ncol=3, columnspacing=1.4, fontsize=10)
    pd.DataFrame(summary).to_csv(OUT / "figure-exp1-2-summary.csv", index=False, encoding="utf-8-sig")
    export(figure, "figure-exp1-2")
    svg = ET.parse(OUT / "figure-exp1-2.svg")
    counts = {}
    for group in svg.iter():
        identifier = group.attrib.get("id", "")
        if identifier.startswith("mc_points_"):
            counts[identifier] = sum(element.tag.endswith("}use") for element in group.iter())
    assert len(counts) == 9 and set(counts.values()) == {20}, counts
    return {"groups": 9, "observations": 180, "repeats_per_observation": 5, "svg_scatter_counts": counts}


def make_figures(fixed_records, trajectories):
    style()
    record = next(record for record in fixed_records["M"] if record["method"] == "SD-TV-UOT")
    first, second = evolution(record), mc_deviation(trajectories)
    captions = {
        "figure-exp1-1": "Fixed M instance (seed 42), analytical SD-TV-UOT. Panels (a)-(d): active warehouses, the two distinct part inventories, and shared design weights. Panels (e)-(h): regions at t=1,3,5 and after recovery. R is the zero-penalty recovery restricted to the terminal positive-stock set; it is not a reweighting iteration. Colors identify warehouses retained after recovery; other candidate trajectories are gray. Map axes are positions in km. Quality comparisons for the fixed S/M/L instances are reported separately in table-exp1-quality.",
        "figure-exp1-2": "MC trajectory deviation relative to the analytical reference across 20 instances per scale. Each circle is one instance's D, the root-mean-square path discrepancy over five independent MC repeats, time, warehouses including external supply, and both parts, normalized by total demand. There are 20 circles per scale-budget group (180 total), not 100 per-repeat observations. Boxes show the median and interquartile range, with 1.5-IQR whiskers; every observation, including outliers, is drawn. Each instance uses a common horizon equal to the maximum stopping time of its analytical and 15 MC paths, includes the common t=0 initialization, and holds a stopped path at its last pre-recovery value. Extra fixed-instance quality repeats do not enter this horizon. Recovery is excluded. MC samples are fixed within a run and nested across budgets; no point is selected, clipped, or resampled based on its result."}
    write_json(OUT / "figure-captions.json", captions)
    return {"figure-exp1-1": first, "figure-exp1-2": second}
