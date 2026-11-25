#!/usr/bin/env python3
"""Generate all figures and LaTeX tables for the final report and slides."""

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "presentation" / "figures"
TAB = ROOT / "presentation" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

ARCHS = ["mpnn", "mpnn_gru", "hypare"]
LABELS = {"mpnn": "MPNN", "mpnn_gru": "MPNN+GRU", "hypare": "HyPARE"}
SEEDS = [0, 1, 2]
LENGTHS = [16, 24, 32, 48, 64]
COLORS = {
    "mpnn": "#234E7A",
    "mpnn_gru": "#DC7D2D",
    "hypare": "#238787",
    "paper": "#8A8A8A",
    "paper_alt": "#B8B8B8",
    "accent": "#735096",
}

PAPER = {
    "bfs": {
        "MPNN": (99.89, 0.05),
        "PGN": (99.63, 0.29),
    },
    "dfs": {
        "MPNN": (6.54, 0.51),
        "PGN": (8.71, 0.24),
        "GAT": (11.78, 2.04),
        "Memnet": (13.36, 1.61),
        "Deep Sets": (7.44, 0.73),
    },
}


def load(path):
    return json.loads((RES / path).read_text())


def ms(values):
    return statistics.mean(values), statistics.pstdev(values)


def pm(mean, std, digits=2):
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def pct(values):
    return [100.0 * value for value in values]


def bfs_results():
    out = {}
    for arch in ARCHS:
        values = [
            load(f"v5_bfs_{arch}_seed{seed}.json")["runs"][arch][str(seed)]["test_ood"]
            for seed in SEEDS
        ]
        mean, std = ms(values)
        out[arch] = {"values": pct(values), "mean": 100 * mean, "std": 100 * std}
    return out


def dfs_controlled_results():
    out = {}
    seeds12 = load("v4_dfs_seeds12.json")["runs"]
    for arch in ARCHS:
        seed0 = load(f"v4_dfs_{arch}_seed0.json")["runs"][arch]["0"]["test_ood"]
        values = [seed0, seeds12[arch]["1"]["test_ood"], seeds12[arch]["2"]["test_ood"]]
        mean, std = ms(values)
        out[arch] = {"values": pct(values), "mean": 100 * mean, "std": 100 * std}
    return out


def dfs_official_results():
    out = {}
    for label, prefix in [
        ("official", "v5_dfs_official"),
        ("official_ln", "v5_dfs_official_ln"),
    ]:
        values = [
            load(f"{prefix}_seed{seed}.json")["runs"]["mpnn"][str(seed)]["test_ood"]
            for seed in SEEDS
        ]
        mean, std = ms(values)
        out[label] = {"values": pct(values), "mean": 100 * mean, "std": 100 * std}
    return out


def diagnostics():
    data = load("v5_dfs_diagnostics.json")
    length = {}
    for size in LENGTHS:
        length[size] = {}
        for arch in ARCHS:
            values = [
                data["length_sweep"][str(size)][f"{arch}/seed{seed}"]["output_accuracy"]
                for seed in SEEDS
            ]
            mean, std = ms(values)
            length[size][arch] = {
                "values": pct(values),
                "mean": 100 * mean,
                "std": 100 * std,
            }
    extended = {}
    for arch in ARCHS:
        values = [
            data["extended"][f"{arch}/seed{seed}"]["output_accuracy"]
            for seed in SEEDS
        ]
        mean, std = ms(values)
        extended[arch] = {
            "values": pct(values),
            "mean": 100 * mean,
            "std": 100 * std,
        }
    return data, length, extended


def interpolate(values, points=50):
    if len(values) < 2:
        return np.full(points, float(values[0]) if values else np.nan)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, points)
    return np.interp(target, source, np.asarray(values, dtype=float))


def gate_and_hint_curves(data, length, hint_lengths):
    gate = {}
    for size in length:
        per_seed = [
            interpolate(
                data["length_sweep"][str(size)][f"hypare/seed{seed}"]["gate_trajectory"]
            )
            for seed in SEEDS
        ]
        arr = np.stack(per_seed)
        gate[size] = (arr.mean(axis=0), arr.std(axis=0))
    hint = {}
    for size in hint_lengths:
        per_seed = [
            interpolate(
                data["length_sweep"][str(size)][f"hypare/seed{seed}"]["hints"]["pi_h"][
                    "per_step_accuracy"
                ]
            )
            for seed in SEEDS
        ]
        arr = np.stack(per_seed)
        hint[size] = (arr.mean(axis=0), arr.std(axis=0))
    return gate, hint


def style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
        }
    )


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


def plot_bfs(bfs):
    style()
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    labels = [LABELS[arch] for arch in ARCHS]
    means = [bfs[arch]["mean"] for arch in ARCHS]
    stds = [bfs[arch]["std"] for arch in ARCHS]
    colors = [COLORS[arch] for arch in ARCHS]
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="black", linewidth=0.6)
    for xi, value, std in zip(x, means, stds):
        ax.text(xi, value + std + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(PAPER["bfs"]["MPNN"][0], color=COLORS["paper"], linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(99.2, 100.08)
    ax.set_ylabel("BFS OOD accuracy (%)")
    ax.set_title("BFS, 64-node test, 3 seeds")
    save(fig, "fig_bfs_reproduction")


def plot_dfs(controlled, official):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.0, 1.35]})

    ax = axes[0]
    labels = [LABELS[arch] for arch in ARCHS]
    means = [controlled[arch]["mean"] for arch in ARCHS]
    stds = [controlled[arch]["std"] for arch in ARCHS]
    colors = [COLORS[arch] for arch in ARCHS]
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="black", linewidth=0.6)
    rng = np.random.default_rng(0)
    for xi, arch in zip(x, ARCHS):
        jitter = rng.uniform(-0.10, 0.10, len(controlled[arch]["values"]))
        ax.scatter(
            xi + jitter,
            controlled[arch]["values"],
            s=12,
            color="black",
            alpha=0.65,
            zorder=3,
            linewidths=0,
        )
        ax.text(
            xi,
            controlled[arch]["mean"] + controlled[arch]["std"] + 1.2,
            f"{controlled[arch]['mean']:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylim(0, 36)
    ax.set_ylabel("DFS OOD accuracy (%)")
    ax.set_title("Matched protocol (ours)")

    ax = axes[1]
    names = list(PAPER["dfs"].keys())
    means = [PAPER["dfs"][name][0] for name in names]
    stds = [PAPER["dfs"][name][1] for name in names]
    x = np.arange(len(names))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=3,
        color=COLORS["paper_alt"],
        edgecolor="black",
        linewidth=0.6,
    )
    for xi, value, std in zip(x, means, stds):
        ax.text(xi, value + std + 0.8, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylim(0, 36)
    ax.set_title("Published context (CLRS paper)")
    fig.suptitle("DFS out-of-distribution accuracy (64 nodes)", fontsize=10, y=1.02)
    fig.text(
        0.5,
        -0.06,
        "Paper numbers use a different implementation; our MPNN baseline is stronger. Context only.",
        ha="center",
        fontsize=7,
        color="#555555",
    )
    save(fig, "fig_dfs_ood")


def plot_length(length, extended):
    style()
    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    for arch in ARCHS:
        means = np.array([length[size][arch]["mean"] for size in LENGTHS])
        stds = np.array([length[size][arch]["std"] for size in LENGTHS])
        ax.plot(LENGTHS, means, marker="o", linewidth=1.8, color=COLORS[arch], label=LABELS[arch])
        ax.fill_between(LENGTHS, means - stds, means + stds, color=COLORS[arch], alpha=0.14)
        ax.annotate(
            f"{means[-1]:.1f}",
            (LENGTHS[-1], means[-1]),
            textcoords="offset points",
            xytext=(6, -2),
            fontsize=7,
            color=COLORS[arch],
        )
    ax.set_xticks(LENGTHS)
    ax.set_xlabel("Test graph size (nodes)")
    ax.set_ylabel("OOD output accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("DFS length generalization (3 seeds)")
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        18,
        8,
        "Train size: 16 nodes",
        fontsize=7,
        color="#555555",
    )
    save(fig, "fig_length_sweep")


def plot_gate(gate, hint):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    xs = np.linspace(0, 100, 50)

    ax = axes[0]
    for size in [16, 32, 64]:
        mean, std = gate[size]
        color = {16: COLORS["mpnn"], 32: COLORS["mpnn_gru"], 64: COLORS["hypare"]}[size]
        ax.plot(xs, mean, linewidth=1.8, color=color, label=f"{size} nodes")
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.13)
    ax.set_xlabel("Rollout progress (%)")
    ax.set_ylabel("Gate value $\\alpha$")
    ax.set_ylim(0, 0.65)
    ax.set_title("Recurrence gate collapses over the rollout")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    for size in [32, 48, 64]:
        mean, std = hint[size]
        color = {32: COLORS["mpnn"], 48: COLORS["mpnn_gru"], 64: COLORS["hypare"]}[size]
        ax.plot(xs, 100 * mean, linewidth=1.8, color=color, label=f"{size} nodes")
        ax.fill_between(xs, 100 * (mean - std), 100 * (mean + std), color=color, alpha=0.13)
    ax.set_xlabel("Rollout progress (%)")
    ax.set_ylabel("$\\pi_h$ hint accuracy (%)")
    ax.set_ylim(30, 103)
    ax.set_title("Errors compound late in the rollout")
    ax.legend(frameon=False, fontsize=7)

    save(fig, "fig_gate_diagnosis")


def plot_gate_vertical(gate, hint):
    style()
    fig, axes = plt.subplots(2, 1, figsize=(3.6, 5.4))
    xs = np.linspace(0, 100, 50)

    ax = axes[0]
    for size in [16, 32, 64]:
        mean, std = gate[size]
        color = {16: COLORS["mpnn"], 32: COLORS["mpnn_gru"], 64: COLORS["hypare"]}[size]
        ax.plot(xs, mean, linewidth=1.6, color=color, label=f"{size} nodes")
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.13)
    ax.set_xlabel("Rollout progress (%)")
    ax.set_ylabel("Gate value $\\alpha$")
    ax.set_ylim(0, 0.65)
    ax.set_title("Recurrence gate collapses over the rollout")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    for size in [32, 48, 64]:
        mean, std = hint[size]
        color = {32: COLORS["mpnn"], 48: COLORS["mpnn_gru"], 64: COLORS["hypare"]}[size]
        ax.plot(xs, 100 * mean, linewidth=1.6, color=color, label=f"{size} nodes")
        ax.fill_between(xs, 100 * (mean - std), 100 * (mean + std), color=color, alpha=0.13)
    ax.set_xlabel("Rollout progress (%)")
    ax.set_ylabel("$\\pi_h$ hint accuracy (%)")
    ax.set_ylim(30, 103)
    ax.set_title("Errors compound late in the rollout")
    ax.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    save(fig, "fig_gate_diagnosis_vertical")


def write_table_bfs(bfs):
    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Model & BFS OOD (\\%) & Seeds \\\\",
        "\\midrule",
    ]
    for arch in ARCHS:
        lines.append(
            f"{LABELS[arch]} (ours) & {pm(bfs[arch]['mean'], bfs[arch]['std'])} & "
            f"{', '.join(f'{v:.2f}' for v in bfs[arch]['values'])} \\\\"
        )
    lines += [
        "\\addlinespace",
        "\\multicolumn{3}{l}{\\emph{CLRS paper (context only)}} \\\\",
    ]
    for name in ["MPNN", "PGN"]:
        mean, std = PAPER["bfs"][name]
        lines.append(f"{name} & {pm(mean, std)} & -- \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "bfs.tex").write_text("\n".join(lines) + "\n")


def write_table_dfs(controlled, official):
    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Model & DFS OOD (\\%) & Seeds \\\\",
        "\\midrule",
        "\\multicolumn{3}{l}{\\emph{Matched protocol (ours, 3 seeds)}} \\\\",
    ]
    for arch in ARCHS:
        lines.append(
            f"{LABELS[arch]} & {pm(controlled[arch]['mean'], controlled[arch]['std'])} & "
            f"{', '.join(f'{v:.2f}' for v in controlled[arch]['values'])} \\\\"
        )
    lines += [
        "\\addlinespace",
        "\\multicolumn{3}{l}{\\emph{Published context (CLRS paper, different implementation)}} \\\\",
    ]
    for name, (mean, std) in PAPER["dfs"].items():
        lines.append(f"{name} & {pm(mean, std)} & -- \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "dfs.tex").write_text("\n".join(lines) + "\n")

    lines = [
        "\\begin{tabular}{lc}",
        "\\toprule",
        "MPNN configuration & DFS OOD (\\%) \\\\",
        "\\midrule",
        f"Original (position aug. + LN) & {pm(controlled['mpnn']['mean'], controlled['mpnn']['std'])} \\\\",
        f"Official-style (no position aug., no LN) & {pm(official['official']['mean'], official['official']['std'])} \\\\",
        f"Official-style (no position aug., LN) & {pm(official['official_ln']['mean'], official['official_ln']['std'])} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (TAB / "dfs_official.tex").write_text("\n".join(lines) + "\n")


def write_table_length(length):
    lines = [
        "\\begin{tabular}{rccc}",
        "\\toprule",
        "Nodes & MPNN & MPNN+GRU & HyPARE \\\\",
        "\\midrule",
    ]
    for size in LENGTHS:
        cells = [pm(length[size][arch]["mean"], length[size][arch]["std"]) for arch in ARCHS]
        lines.append(f"{size} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "length_sweep.tex").write_text("\n".join(lines) + "\n")


def write_summary(bfs, controlled, official, length, extended):
    summary = {
        "bfs": bfs,
        "dfs_controlled": controlled,
        "dfs_official": official,
        "dfs_length_sweep": length,
        "dfs_extended_256": extended,
        "paper": PAPER,
    }
    (ROOT / "presentation" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    data, length, extended = diagnostics()
    bfs = bfs_results()
    controlled = dfs_controlled_results()
    official = dfs_official_results()
    gate, hint = gate_and_hint_curves(data, [16, 32, 64], [32, 48, 64])

    plot_bfs(bfs)
    plot_dfs(controlled, official)
    plot_length(length, extended)
    plot_gate(gate, hint)
    plot_gate_vertical(gate, hint)
    write_table_bfs(bfs)
    write_table_dfs(controlled, official)
    write_table_length(length)
    write_summary(bfs, controlled, official, length, extended)

    print("=== BFS ===")
    for arch in ARCHS:
        print(arch, f"{bfs[arch]['mean']:.2f} +/- {bfs[arch]['std']:.2f}", bfs[arch]["values"])
    print("=== DFS controlled ===")
    for arch in ARCHS:
        print(arch, f"{controlled[arch]['mean']:.2f} +/- {controlled[arch]['std']:.2f}", controlled[arch]["values"])
    print("=== DFS official ===")
    for label, value in official.items():
        print(label, f"{value['mean']:.2f} +/- {value['std']:.2f}", value["values"])
    print("=== length sweep ===")
    for size in LENGTHS:
        print(size, {arch: round(length[size][arch]["mean"], 2) for arch in ARCHS})
    print("=== extended 256 @ 64 ===")
    for arch in ARCHS:
        print(arch, f"{extended[arch]['mean']:.2f} +/- {extended[arch]['std']:.2f}")
    print(f"figures -> {FIG}")
    print(f"tables  -> {TAB}")


if __name__ == "__main__":
    main()
