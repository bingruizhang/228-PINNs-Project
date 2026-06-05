from __future__ import annotations

import math

import torch
from torch import nn


def normalize_coords(coords: torch.Tensor) -> torch.Tensor:
    """Map x,y in [0, 2pi] and t in [0, 1] to roughly [-1, 1]."""

    x = coords[:, 0:1] / math.pi - 1.0
    y = coords[:, 1:2] / math.pi - 1.0
    t = 2.0 * coords[:, 2:3] - 1.0
    return torch.cat([x, y, t], dim=1)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 3,
        width: int = 64,
        depth: int = 5,
        activation: str = "tanh",
        normalize_input: bool = True,
    ):
        super().__init__()
        self.normalize_input = normalize_input
        act = {"tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
        layers: list[nn.Module] = [nn.Linear(in_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = normalize_coords(coords) if self.normalize_input else coords
        return self.net(x)


class FourierMLP(nn.Module):
    def __init__(self, num_frequencies: int = 8, width: int = 96, depth: int = 5, sigma: float = 2.0):
        super().__init__()
        freqs = sigma * torch.arange(1, num_frequencies + 1).float()
        self.register_buffer("freqs", freqs)
        in_dim = 3 + 2 * 3 * num_frequencies
        self.mlp = MLP(in_dim=in_dim, out_dim=3, width=width, depth=depth, activation="gelu", normalize_input=False)

    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        coords = normalize_coords(coords)
        z = math.pi * coords[..., None] * self.freqs
        features = torch.cat([torch.sin(z), torch.cos(z)], dim=-1).flatten(1)
        return torch.cat([coords, features], dim=1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encode(coords))


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, width))
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ResidualMLP(nn.Module):
    def __init__(self, width: int = 96, blocks: int = 4):
        super().__init__()
        self.in_layer = nn.Sequential(nn.Linear(3, width), nn.Tanh())
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.out_layer = nn.Linear(width, 3)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.blocks(self.in_layer(normalize_coords(coords))))


class SineLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, omega_0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            bound = 1.0 / in_dim if is_first else math.sqrt(6.0 / in_dim) / omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    def __init__(self, width: int = 96, depth: int = 5, omega_0: float = 30.0):
        super().__init__()
        layers: list[nn.Module] = [SineLayer(3, width, omega_0=omega_0, is_first=True)]
        for _ in range(depth - 1):
            layers.append(SineLayer(width, width, omega_0=omega_0))
        layers.append(nn.Linear(width, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(normalize_coords(coords))


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        scale = 1.0 / (in_channels * out_channels)
        self.modes = modes
        self.weights = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, nx, ny = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batch, self.weights.shape[1], nx, ny // 2 + 1, device=x.device, dtype=torch.cfloat)
        mx = min(self.modes, nx)
        my = min(self.modes, ny // 2 + 1)
        out_ft[:, :, :mx, :my] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :mx, :my], self.weights[:, :, :mx, :my])
        return torch.fft.irfft2(out_ft, s=(nx, ny))


class TinyFNO2d(nn.Module):
    """Small grid-conditioned FNO baseline for whole-field prediction."""

    def __init__(self, width: int = 32, modes: int = 12, layers: int = 4):
        super().__init__()
        self.lift = nn.Conv2d(3, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, 64, 1), nn.GELU(), nn.Conv2d(64, 3, 1))

    def forward(self, grid_channels: torch.Tensor) -> torch.Tensor:
        x = self.lift(grid_channels)
        for spec, local in zip(self.spec, self.local):
            x = torch.nn.functional.gelu(spec(x) + local(x))
        return self.proj(x)


def make_model(name: str, width: int = 96, depth: int = 5) -> nn.Module:
    name = name.lower()
    if name == "fcn":
        return MLP(width=width, depth=depth, activation="tanh")
    if name == "deep_fcn":
        return MLP(width=width, depth=max(depth, 8), activation="tanh")
    if name == "fourier":
        return FourierMLP(width=width, depth=depth)
    if name == "siren":
        return SIREN(width=width, depth=depth)
    if name == "residual":
        return ResidualMLP(width=width, blocks=max(2, depth // 2))
    if name == "fno":
        return TinyFNO2d(width=max(16, width // 3), modes=12, layers=4)
    raise ValueError(f"Unknown model: {name}")
