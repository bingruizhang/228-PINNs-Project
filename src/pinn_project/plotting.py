from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .data import Domain, grid_coords, taylor_green_solution


def plot_slice(model, out_path: Path, device: torch.device, t_value: float = 0.5, nx: int = 64, ny: int = 64) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    domain = Domain()
    coords = grid_coords(nx, ny, t_value, domain, device)
    with torch.no_grad():
        pred = model(coords).detach().cpu()
        true = taylor_green_solution(coords).detach().cpu()
    pred_img = pred.reshape(ny, nx, 3)
    true_img = true.reshape(ny, nx, 3)
    err_img = (pred_img - true_img).abs()

    fig, axes = plt.subplots(3, 3, figsize=(10, 9), constrained_layout=True)
    labels = ["u", "v", "p"]
    for col, label in enumerate(labels):
        for row, img in enumerate([true_img, pred_img, err_img]):
            ax = axes[row, col]
            im = ax.imshow(img[:, :, col], origin="lower", cmap="coolwarm")
            ax.set_title(["truth", "prediction", "abs error"][row] + f" {label}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.75)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_history(history: list[dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [row["step"] for row in history]
    for key in ["loss", "data_loss", "pde_loss", "continuity_loss"]:
        vals = [row.get(key, float("nan")) for row in history]
        ax.plot(xs, vals, label=key)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
