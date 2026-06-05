from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Domain:
    x_min: float = 0.0
    x_max: float = 2.0 * math.pi
    y_min: float = 0.0
    y_max: float = 2.0 * math.pi
    t_min: float = 0.0
    t_max: float = 1.0


def taylor_green_solution(coords: torch.Tensor, nu: float = 0.01) -> torch.Tensor:
    """Analytic incompressible 2D Navier-Stokes solution: [u, v, p]."""

    x = coords[:, 0:1]
    y = coords[:, 1:2]
    t = coords[:, 2:3]
    decay_v = torch.exp(-2.0 * nu * t)
    decay_p = torch.exp(-4.0 * nu * t)
    u = -torch.cos(x) * torch.sin(y) * decay_v
    v = torch.sin(x) * torch.cos(y) * decay_v
    p = -0.25 * (torch.cos(2.0 * x) + torch.cos(2.0 * y)) * decay_p
    return torch.cat([u, v, p], dim=1)


def sample_coords(n: int, domain: Domain, device: torch.device, seed: int | None = None) -> torch.Tensor:
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    r = torch.rand((n, 3), generator=gen, device=device)
    x = domain.x_min + (domain.x_max - domain.x_min) * r[:, 0:1]
    y = domain.y_min + (domain.y_max - domain.y_min) * r[:, 1:2]
    t = domain.t_min + (domain.t_max - domain.t_min) * r[:, 2:3]
    return torch.cat([x, y, t], dim=1)


def grid_coords(nx: int, ny: int, t_value: float, domain: Domain, device: torch.device) -> torch.Tensor:
    xs = torch.linspace(domain.x_min, domain.x_max, nx, device=device)
    ys = torch.linspace(domain.y_min, domain.y_max, ny, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    tt = torch.full_like(xx, float(t_value))
    return torch.stack([xx, yy, tt], dim=-1).reshape(-1, 3)


def add_noise(values: torch.Tensor, noise_std: float, seed: int | None = None) -> torch.Tensor:
    if noise_std <= 0.0:
        return values
    gen = torch.Generator(device=values.device)
    if seed is not None:
        gen.manual_seed(seed)
    scale = values.std(dim=0, keepdim=True).clamp_min(1e-6)
    return values + noise_std * scale * torch.randn(values.shape, generator=gen, device=values.device)

