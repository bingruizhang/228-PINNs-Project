from __future__ import annotations

import torch


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def navier_stokes_residual(
    coords: torch.Tensor,
    pred: torch.Tensor,
    lambda_1: torch.Tensor,
    lambda_2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p = pred[:, 2:3]

    du = grad(u, coords)
    dv = grad(v, coords)
    dp = grad(p, coords)
    u_x, u_y, u_t = du[:, 0:1], du[:, 1:2], du[:, 2:3]
    v_x, v_y, v_t = dv[:, 0:1], dv[:, 1:2], dv[:, 2:3]
    p_x, p_y = dp[:, 0:1], dp[:, 1:2]

    u_xx = grad(u_x, coords)[:, 0:1]
    u_yy = grad(u_y, coords)[:, 1:2]
    v_xx = grad(v_x, coords)[:, 0:1]
    v_yy = grad(v_y, coords)[:, 1:2]

    f_u = u_t + lambda_1 * (u * u_x + v * u_y) + p_x - lambda_2 * (u_xx + u_yy)
    f_v = v_t + lambda_1 * (u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)
    continuity = u_x + v_y
    return f_u, f_v, continuity

