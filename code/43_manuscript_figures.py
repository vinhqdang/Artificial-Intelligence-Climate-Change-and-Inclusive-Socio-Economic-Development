"""Build vector figures for the manuscript from the replication data."""

import argparse
import hashlib
import json
from importlib import import_module
from pathlib import Path
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyBboxPatch, Patch, PathPatch
from matplotlib.path import Path as PlotPath
from matplotlib.ticker import FuncFormatter, PercentFormatter

from ewdrpl.policy import topk_policy

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "figures"
BOUNDARY_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "v5.1.2/geojson/ne_50m_admin_0_countries.geojson"
)
BOUNDARY_SHA256 = "3e458fc036ad0a66411f2c1e6cac49c5d7bfb81cb1123bc513b22511a2b7fdeb"
BLUE, GOLD, TEAL, PURPLE, INK = "#276391", "#c37b24", "#237c70", "#795583", "#253445"
MUTED, GRID, OUT = "#697888", "#e3e9ef", "#e8eaec"
RESEARCH, POOREST, AGRICULTURAL = "Research-first", "Poorest-first", "Most-agricultural-first"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": INK, "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.fontsize": 7, "axes.axisbelow": True, "pdf.fonttype": 42,
    })


def save(fig, name):
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.045)
    plt.close(fig)


def study_inputs():
    app = import_module("31_application_agri")
    observations = app.load_panel()
    estimation_sample, _ = app.make_frame(observations)
    countries = app.policy_country_frame(estimation_sample, observations)
    selection = pd.DataFrame({
        RESEARCH: topk_policy(countries["ln_kpc_lag"], app.K_MAIN),
        POOREST: topk_policy(-countries["ln_gdppc"], app.K_MAIN),
        AGRICULTURAL: topk_policy(countries["agri"], app.K_MAIN),
    }, index=countries.index)
    exposure = countries["pop"] * countries["agri"] * countries["s_bar"]
    shares = app.exposure_shares(countries, app.K_MAIN).set_index("rule")
    percent = 100 * exposure / exposure.sum()
    np.testing.assert_allclose(
        selection.to_numpy(dtype=float).T @ percent.to_numpy(),
        shares["exposure_eps0_pct"].to_numpy(), atol=1e-9,
    )
    assert len(countries) == 126 and selection.sum().eq(37).all()
    return percent, selection, shares


def boundaries(path):
    if not path.is_file():
        with urlopen(BOUNDARY_URL, timeout=35) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != BOUNDARY_SHA256:
            raise ValueError("Natural Earth download does not match the pinned release")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BOUNDARY_SHA256:
        raise ValueError("expected Natural Earth 1:50m countries GeoJSON v5.1.2")
    return json.loads(raw)["features"]


def equal_earth(points):
    """Equal Earth (EPSG:8857), with the radius scaled out."""
    lon, lat = np.deg2rad(np.asarray(points).T)
    t = np.arcsin(np.sqrt(3) / 2 * np.sin(lat))
    a, b, c, d = 1.340264, -0.081106, 0.000893, 0.003796
    x = 2 * np.sqrt(3) * lon * np.cos(t) / (3 * (a + 3*b*t**2 + 7*c*t**6 + 9*d*t**8))
    y = t * (a + b*t**2 + c*t**6 + d*t**8)
    return np.column_stack([x, y])


def country_paths(features, sample):
    paths = []
    for feature in features:
        props = feature["properties"]
        if props["ADM0_A3"] == "ATA":
            continue
        alternatives = (props["ISO_A3"], props["ISO_A3_EH"], props["ADM0_A3"], props["WB_A3"])
        iso = next((code for code in alternatives if code in sample), None)
        geometry = feature["geometry"]
        polygons = geometry["coordinates"]
        if geometry["type"] == "Polygon":
            polygons = [polygons]
        for polygon in polygons:
            rings = []
            for points in polygon:
                path = equal_earth(points)
                rings.append(PlotPath(path, [PlotPath.MOVETO]
                                      + [PlotPath.LINETO] * (len(path) - 2)
                                      + [PlotPath.CLOSEPOLY]))
            paths.append((iso, PlotPath.make_compound_path(*rings)))
    missing = set(sample) - {iso for iso, _ in paths}
    if missing:
        raise ValueError(f"no Natural Earth geometry for policy-sample countries: {sorted(missing)}")
    return paths


def world_map(axis, paths, shades):
    coll = PatchCollection([PathPatch(p) for _, p in paths],
                           facecolors=[shades.get(iso, OUT) for iso, _ in paths],
                           edgecolors="white", linewidths=0.26, rasterized=False)
    axis.add_collection(coll)
    axis.set(xlim=(-2.74, 2.74), ylim=(-1.08, 1.34), aspect="equal")
    axis.set_axis_off()


def plot_maps(percent, selection, features):
    paths = country_paths(features, percent.index)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.55), layout="constrained")
    bins = [0, 0.05, 0.2, 0.5, 2, 100]
    palette = ListedColormap(["#e0ecf6", "#b8d2e9", "#80aecf", "#417ea8", "#174668"])
    norm = BoundaryNorm(bins, palette.N)
    world_map(axes[0], paths, dict(zip(percent.index, palette(norm(percent)))))
    axes[0].set_title("A   Country share of measured agricultural heat exposure", loc="left", weight="bold")
    top_legend = [
        Patch(color=color, label=label)
        for color, label in zip(palette.colors, ["<0.05%", "0.05–0.2%", "0.2–0.5%", "0.5–2%", "≥2%"])
    ] + [Patch(color=OUT, label="Outside sample")]
    axes[0].legend(handles=top_legend, frameon=False, ncols=6, loc="upper center",
                   bbox_to_anchor=(0.5, 0.022), handlelength=1.0, columnspacing=1.1)

    group = selection[RESEARCH].astype(int) + 2 * selection[POOREST].astype(int)
    groups = {0: ("Neither", "#d4dce5"), 1: ("Research only", BLUE),
              2: ("Poorest only", GOLD), 3: ("Both", PURPLE)}
    world_map(axes[1], paths, {iso: groups[key][1] for iso, key in group.items()})
    axes[1].set_title("B   Countries selected under two rules with the same 37-country budget", loc="left", weight="bold")
    counts = group.value_counts()
    labels = [Patch(color=color, label=f"{label} ({counts[key]})")
              for key, (label, color) in groups.items() if key in counts]
    labels.append(Patch(color=OUT, label="Outside sample"))
    axes[1].legend(handles=labels, frameon=False, ncols=5, loc="upper center",
                   bbox_to_anchor=(0.5, 0.023), handlelength=1.0, columnspacing=1.2)
    save(fig, "allocation_maps")


def plot_coverage(shares):
    fig, ax = plt.subplots(figsize=(6.7, 2.65), layout="constrained")
    labels = [RESEARCH, POOREST, AGRICULTURAL]
    specs = {
        "exposure_eps0_pct": ("Equal welfare weight per person", BLUE, "o"),
        "bottom_eps1_pct": ("Poverty weight, bottom-first exposure", GOLD, "s"),
        "uniform_eps1_pct": ("Poverty weight, uniform exposure", TEAL, "D"),
    }
    y = np.array([3, 2, 1])
    for i, (col, (label, color, marker)) in enumerate(specs.items()):
        numbers = shares.loc[labels, col].to_numpy()
        pos = y + (1 - i) * 0.16
        ax.scatter(numbers, pos, color=color, marker=marker, s=25, label=label, zorder=3)
        for value, row in zip(numbers, pos):
            ax.annotate(f"{value:.1f}", (value, row), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=7, color=color)
    baseline = 100 * 37 / 126
    ax.axvline(baseline, color=MUTED, linestyle=(0, (3, 3)), linewidth=1)
    ax.text(baseline + 1.3, 3.42, "Random allocation: 29.4%", color=MUTED, va="center", fontsize=7)
    ax.set(xlim=(0, 95), ylim=(0.61, 3.66), yticks=y, yticklabels=labels,
           xlabel="Share of sample exposure covered")
    ax.set_xticks(np.arange(0, 91, 15))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(axis="x", color=GRID)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncols=3, columnspacing=1.25)
    save(fig, "exposure_coverage")


def diagram_box(ax, x, y, w, h, heading, detail, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                               facecolor=color, edgecolor="none"))
    ax.text(x + 0.023, y + h - 0.085, heading, weight="bold", va="top", fontsize=8.7)
    ax.text(x + 0.023, y + h - 0.22, detail, va="top", linespacing=1.43, fontsize=8)


def plot_algorithm():
    fig = plt.figure(figsize=(7, 5.1), layout="constrained")
    layout = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.32])
    flow = fig.add_subplot(layout[0, :])
    flow.set(xlim=(0, 1), ylim=(0, 1))
    flow.axis("off")
    flow.set_title("A   Join additions and removals with the same exchange count", weight="bold", loc="left")
    diagram_box(flow, 0.01, 0.20, 0.27, 0.60, "Addition table", "Count $m$ and signed grid sum\nBest reward and its witness\nExact residual limits", "#e6eff7")
    diagram_box(flow, 0.36, 0.20, 0.27, 0.60, "Removal table", "Count $m$ and signed grid sum\nBest reward and its witness\nExact residual limits", "#faefdf")
    diagram_box(flow, 0.71, 0.20, 0.27, 0.60, "Joined cell",
                "Combine rewards and limits\nScore a feasible witness\nBound the cell's best value", "#e2f1ec")
    for x in (0.32, 0.67):
        flow.text(x, 0.50, "+" if x < 0.5 else "→", ha="center", va="center", fontsize=14, color=MUTED)

    interval = fig.add_subplot(layout[1, 0])
    interval.set(xlim=(0.08, 0.15), ylim=(0, 1.0))
    interval.axis("off")
    interval.set_title("B   One cell, two feasible portfolios", weight="bold", loc="left")
    interval.plot([0.10, 0.13], [0.62, 0.62], color=TEAL, linewidth=7, solid_capstyle="round")
    interval.scatter([0.10, 0.13], [0.62, 0.62], s=28, color=[GOLD, BLUE], zorder=3)
    interval.text(0.10, 0.50, "0.10 load", ha="center", color=GOLD)
    interval.text(0.13, 0.50, "0.13 load", ha="center", color=BLUE)
    interval.text(0.10, 0.73, "0.28 reward", ha="center", color=GOLD)
    interval.text(0.13, 0.73, "0.30 reward", ha="center", color=BLUE)
    interval.text(0.082, 0.29, "Feasible witness:  0.30 − 0.13 = 0.17", fontsize=8)
    interval.text(0.082, 0.12, "Cell upper bound:  0.30 − 0.10 = 0.20", fontsize=8, weight="bold")

    tilt = fig.add_subplot(layout[1, 1])
    tilt.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    tilt.axis("off")
    tilt.set_title("C   Account for omitted risk", weight="bold", loc="left")
    start = (0.14, 0.15)
    split = (0.77, 0.15)
    end = (0.77, 0.76)
    for origin, tip, color in ((start, split, BLUE), (split, end, GOLD), (start, end, TEAL)):
        tilt.annotate("", xy=tip, xytext=origin,
                      arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.8})
    tilt.text(0.45, 0.045, "Risk kept in grid", color=BLUE, ha="center")
    tilt.text(0.82, 0.46, "Omitted risk", color=GOLD, va="center", rotation=90)
    tilt.text(0.33, 0.54, "Full risk", color=TEAL, rotation=43)
    tilt.text(0.16, 0.87, "Use this omitted direction in the next pass.", fontsize=7.5)
    save(fig, "exchange_grid_algorithm")


def plot_heldout():
    df = pd.read_csv(ROOT / "results/ellipsoidal_allocation/application.csv")
    methods = ["Robust swaps", "Pooled mean", "Ellipsoidal"]
    colors = [GOLD, BLUE, TEAL]
    labels = ["Threshold swaps", "Pooled-mean ranking", "Ellipsoidal portfolio"]
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6.8, 2.5), layout="constrained")
    for ax, (design, rows) in zip(axes, df[df["evaluation"] == "DR 0.10"].groupby("design", sort=False)):
        points = rows.set_index("method").loc[methods]
        ax.set_title("Hold out countries" if design == "Country folds" else "Hold out whole regions",
                     loc="left", weight="bold")
        ax.axvline(0, color=MUTED, lw=1, linestyle=(0, (3, 3)))
        for y, (i, row) in enumerate(zip(colors, points.itertuples())):
            ax.errorbar(row.gain, 2 - y, xerr=[[row.gain - row.lower], [row.upper - row.gain]],
                        fmt="o", color=i, capsize=3, markersize=5, linewidth=1.45)
            ax.annotate(f"{row.gain:+.3f}" if row.gain else "0.000", (row.gain, 2 - y),
                        xytext=(0, 8), textcoords="offset points", fontsize=7, ha="center", color=i)
        ax.set(ylim=(-0.38, 2.43), yticks=[2, 1, 0], yticklabels=labels, xlim=(-1.55, 1.65))
        ax.grid(axis="x", color=GRID)
    fig.supxlabel("Estimated welfare gain relative to the exposure reference", fontsize=8.5)
    save(fig, "heldout_welfare")


def plot_simulation():
    table = pd.read_csv(ROOT / "results/ellipsoidal_allocation/simulation.csv")
    methods = ["Robust swaps", "Pooled mean", "Ellipsoidal"]
    designs = ["Moderate overlap", "Weak overlap", "Homogeneous effects"]
    marks = ["s", "D", "o"]
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(7.1, 2.75), layout="constrained")
    for ax, col, label in zip(axes, ["loss_rate", "gain"], ["Runs worse than reference", "Mean welfare gain"]):
        for i, (method, color, marker) in enumerate(zip(methods, [GOLD, BLUE, TEAL], marks)):
            data = table[table["method"] == method].set_index("design").loc[designs, col]
            y = np.arange(2, -1, -1) + (1 - i) * 0.17
            ax.scatter(data, y, color=color, marker=marker, s=27, zorder=3)
        ax.axvline(0, color=MUTED, lw=0.8, linestyle=(0, (3, 3)))
        ax.set(xlabel=label, yticks=[2, 1, 0], yticklabels=designs, ylim=(-0.35, 2.38))
        ax.grid(axis="x", color=GRID)
        if col == "loss_rate":
            ax.set_xlim(-0.035, 1.04)
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        else:
            ax.set_xlim(-0.025, 0.022)
            ax.set_xticks([-0.02, -0.01, 0, 0.01, 0.02])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    legend = [plt.Line2D([], [], linestyle="none", marker=m, color=c, label=l)
              for m, c, l in zip(marks, [GOLD, BLUE, TEAL],
                                 ["Threshold swaps", "Pooled mean", "Ellipsoidal"])]
    axes[-1].legend(handles=legend, frameon=False, loc="upper center",
                    bbox_to_anchor=(0.0, -0.23), ncols=3, columnspacing=1.75)
    save(fig, "simulation_tradeoff")


def plot_solver():
    table = pd.read_csv(ROOT / "results/exchange_grid/summary.csv")
    designs = list(table["design"].unique())
    methods = ["Exchange grid", "Grid without tilt", "Integer cutting planes"]
    labels = ["One-factor lattice", "Two-factor lattice", "Six continuous factors",
              "Agriculture, country folds", "Agriculture, region folds"]
    colors = [TEAL, MUTED, PURPLE]
    markers = ["o", "s", "D"]
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(7.15, 3.45), layout="constrained")
    for ax, col in zip(axes, ["certified", "median_seconds"]):
        for i, (method, color, marker) in enumerate(zip(methods, colors, markers)):
            rows = table[table["method"] == method].set_index("design").loc[designs]
            pos = np.arange(4, -1, -1) + (1 - i) * 0.19
            values = rows[col] / rows["problems"] if col == "certified" else 1000 * rows[col]
            ax.scatter(values, pos, color=color, marker=marker, s=26, zorder=3)
            if col == "certified":
                for x, y, a, b in zip(values, pos, rows["certified"], rows["problems"]):
                    ax.annotate(f"{a}/{b}", (x, y), xytext=(5, 0), textcoords="offset points",
                                va="center", fontsize=6.4, color=color)
        ax.set(yticks=np.arange(4, -1, -1), yticklabels=labels, ylim=(-0.42, 4.42))
        ax.grid(axis="x", color=GRID)
        if col == "certified":
            ax.set(xlim=(-0.08, 1.28), xlabel="Instances certified", xticks=[0, 0.25, 0.5, 0.75, 1])
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        else:
            ax.set_xscale("log")
            ax.set(xlim=(0.5, 700), xlabel="Median solver time (ms, log scale)", xticks=[1, 10, 100])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    legend = [plt.Line2D([], [], linestyle="none", marker=m, color=c, label=l)
              for m, c, l in zip(markers, colors, methods)]
    axes[-1].legend(handles=legend, frameon=False, loc="upper center",
                    bbox_to_anchor=(0.0, -0.20), ncols=3, columnspacing=1.4)
    save(fig, "solver_benchmark")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundaries", type=Path,
                        default=ROOT / "data/raw/natural_earth/ne_50m_admin_0_countries_v5.1.2.geojson")
    args = parser.parse_args()
    style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    percent, selection, shares = study_inputs()
    plot_maps(percent, selection, boundaries(args.boundaries))
    plot_coverage(shares)
    plot_algorithm()
    plot_heldout()
    plot_simulation()
    plot_solver()


if __name__ == "__main__":
    main()
