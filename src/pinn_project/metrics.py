from __future__ import annotations

import torch


def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> float:
    return (torch.linalg.norm(pred - true) / torch.linalg.norm(true).clamp_min(1e-12)).item()


def channel_metrics(pred: torch.Tensor, true: torch.Tensor) -> dict[str, float]:
    names = ["u", "v", "p"]
    return {f"rel_l2_{name}": relative_l2(pred[:, i], true[:, i]) for i, name in enumerate(names)}

