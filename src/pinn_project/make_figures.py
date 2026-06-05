from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in {"", "None", "nan"}:
        return float("nan")
    return float(value)


def plot_architecture(rows: list[dict[str, str]], out_dir: Path) -> None:
    arch = [r for r in rows if r.get("run_name", "").startswith("arch_")]
    if not arch:
        return
    arch.sort(key=lambda r: r["model"])
    labels = [r["model"] for r in arch]
    err = [as_float(r, "rel_l2_u") + as_float(r, "rel_l2_v") for r in arch]
    lam = [as_float(r, "lambda_1_abs_error") for r in arch]
    time = [as_float(r, "train_seconds") for r in arch]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    axes[0].bar(labels, err)
    axes[0].set_title("Velocity relative L2 error")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(labels, lam)
    axes[1].set_title("lambda_1 abs error")
    axes[1].tick_params(axis="x", rotation=30)
    axes[2].scatter(time, err)
    for x, y, label in zip(time, err, labels):
        axes[2].annotate(label, (x, y))
    axes[2].set_xlabel("training seconds")
    axes[2].set_ylabel("velocity relative L2")
    axes[2].set_title("Accuracy-time tradeoff")
    fig.savefig(out_dir / "architecture_summary.png", dpi=180)
    plt.close(fig)


def plot_grouped_lines(rows: list[dict[str, str]], prefix: str, x_key: str, out_path: Path, title: str) -> None:
    subset = [r for r in rows if r.get("run_name", "").startswith(prefix)]
    if not subset:
        return
    models = sorted({r["model"] for r in subset})
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for model in models:
        part = [r for r in subset if r["model"] == model]
        part.sort(key=lambda r: as_float(r, x_key))
        xs = [as_float(r, x_key) for r in part]
        ys = [as_float(r, "rel_l2_u") + as_float(r, "rel_l2_v") for r in part]
        ax.plot(xs, ys, marker="o", label=model)
    ax.set_xlabel(x_key)
    ax.set_ylabel("velocity relative L2")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make final-project summary figures.")
    parser.add_argument("--summary", default="results/final_summary.csv")
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(Path(args.summary))
    plot_architecture(rows, out_dir)
    plot_grouped_lines(rows, "data_", "n_obs", out_dir / "data_scarcity.png", "Data scarcity")
    plot_grouped_lines(rows, "noise_", "noise_std", out_dir / "noise_robustness.png", "Noise robustness")
    plot_grouped_lines(rows, "time_", "train_t_max", out_dir / "time_generalization.png", "Time generalization")
    print(f"Wrote summary figures to {out_dir}")


if __name__ == "__main__":
    main()
