from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .data import Domain, add_noise, grid_coords, sample_coords, taylor_green_solution
from .metrics import channel_metrics, relative_l2
from .models import TinyFNO2d, make_model
from .physics import navier_stokes_residual
from .plotting import plot_history, plot_slice


def device_from_arg(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def write_history(history: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def inverse_softplus(value: float) -> torch.Tensor:
    return torch.log(torch.expm1(torch.tensor(value)).clamp_min(1e-12))


def evaluate_coord_model(model: nn.Module, device: torch.device, nu: float, lambda_1: torch.Tensor, lambda_2: torch.Tensor) -> dict[str, float]:
    domain = Domain()
    metrics: dict[str, float] = {}
    all_pred = []
    all_true = []
    for t_value in [0.25, 0.50, 0.75]:
        coords = grid_coords(72, 72, t_value, domain, device)
        with torch.no_grad():
            pred = model(coords)
            true = taylor_green_solution(coords, nu=nu)
        all_pred.append(pred)
        all_true.append(true)
    pred = torch.cat(all_pred, dim=0)
    true = torch.cat(all_true, dim=0)
    metrics.update(channel_metrics(pred, true))
    metrics["lambda_1"] = float(lambda_1.detach().cpu())
    metrics["lambda_2"] = float(lambda_2.detach().cpu())
    metrics["lambda_1_abs_error"] = abs(metrics["lambda_1"] - 1.0)
    metrics["lambda_2_abs_error"] = abs(metrics["lambda_2"] - nu)

    coords = sample_coords(2048, domain, device, seed=123).requires_grad_(True)
    pred_r = model(coords)
    f_u, f_v, cont = navier_stokes_residual(coords, pred_r, lambda_1, lambda_2)
    metrics["pde_residual_mse"] = float((f_u.square().mean() + f_v.square().mean()).detach().cpu())
    metrics["continuity_mse"] = float(cont.square().mean().detach().cpu())
    return metrics


def train_coord_model(args: argparse.Namespace) -> dict[str, float]:
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = device_from_arg(args.device)
    domain = Domain()
    train_domain = Domain(t_max=args.train_t_max)
    result_dir = Path(args.output_dir) / args.run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    model = make_model(args.model, width=args.width, depth=args.depth).to(device)
    raw_lambda_1 = nn.Parameter(inverse_softplus(args.lambda1_init).to(device))
    raw_lambda_2 = nn.Parameter(inverse_softplus(args.lambda2_init).to(device))
    optimizer = torch.optim.AdamW(list(model.parameters()) + [raw_lambda_1, raw_lambda_2], lr=args.lr, weight_decay=args.weight_decay)

    obs_coords = sample_coords(args.n_obs, train_domain, device, seed=args.seed)
    obs_true = taylor_green_solution(obs_coords, nu=args.nu)
    obs_target = add_noise(obs_true, args.noise_std, seed=args.seed + 17)

    history: list[dict[str, float]] = []
    start = time.perf_counter()
    peak_mem = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        data_idx = torch.randint(0, args.n_obs, (min(args.data_batch, args.n_obs),), device=device)
        data_coords = obs_coords[data_idx]
        data_pred = model(data_coords)
        target = obs_target[data_idx]
        uv_loss = torch.mean((data_pred[:, :2] - target[:, :2]) ** 2)
        p_loss = torch.mean((data_pred[:, 2:3] - target[:, 2:3]) ** 2)
        data_loss = uv_loss + args.p_data_weight * p_loss

        colloc = sample_coords(args.colloc_batch, train_domain, device).requires_grad_(True)
        colloc_pred = model(colloc)
        lambda_1 = F.softplus(raw_lambda_1)
        lambda_2 = F.softplus(raw_lambda_2)
        f_u, f_v, cont = navier_stokes_residual(colloc, colloc_pred, lambda_1, lambda_2)
        pde_loss = f_u.square().mean() + f_v.square().mean()
        continuity_loss = cont.square().mean()
        warmup = 1.0 if args.pde_warmup_steps <= 0 else min(1.0, step / args.pde_warmup_steps)
        loss = data_loss + warmup * args.pde_weight * pde_loss + warmup * args.continuity_weight * continuity_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [raw_lambda_1, raw_lambda_2], max_norm=10.0)
        optimizer.step()

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            if device.type == "cuda":
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(device))
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "data_loss": float(data_loss.detach().cpu()),
                    "pde_loss": float(pde_loss.detach().cpu()),
                    "continuity_loss": float(continuity_loss.detach().cpu()),
                    "lambda_1": float(lambda_1.detach().cpu()),
                    "lambda_2": float(lambda_2.detach().cpu()),
                    "pde_warmup": warmup,
                }
            )

    train_seconds = time.perf_counter() - start
    lambda_1 = F.softplus(raw_lambda_1)
    lambda_2 = F.softplus(raw_lambda_2)
    metrics = evaluate_coord_model(model, device, args.nu, lambda_1, lambda_2)
    metrics.update(
        {
            "model": args.model,
            "run_name": args.run_name,
            "n_obs": args.n_obs,
            "noise_std": args.noise_std,
            "train_t_max": args.train_t_max,
            "steps": args.steps,
            "train_seconds": train_seconds,
            "peak_gpu_mb": peak_mem / (1024**2),
        }
    )

    with (result_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    write_history(history, result_dir / "history.csv")
    plot_history(history, result_dir / "loss_curve.png")
    plot_slice(model, result_dir / "slice_t050.png", device=device)
    torch.save({"model_state": model.state_dict(), "metrics": metrics}, result_dir / "checkpoint.pt")
    return metrics


def fno_input_grid(batch_t: torch.Tensor, nx: int, ny: int, domain: Domain, device: torch.device) -> torch.Tensor:
    xs = torch.linspace(domain.x_min, domain.x_max, nx, device=device)
    ys = torch.linspace(domain.y_min, domain.y_max, ny, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx = xx / domain.x_max
    yy = yy / domain.y_max
    grids = []
    for t in batch_t:
        tt = torch.full_like(xx, t)
        grids.append(torch.stack([xx, yy, tt], dim=0))
    return torch.stack(grids, dim=0)


def fno_target_grid(batch_t: torch.Tensor, nx: int, ny: int, domain: Domain, device: torch.device, nu: float) -> torch.Tensor:
    coords_list = [grid_coords(nx, ny, float(t.item()), domain, device) for t in batch_t]
    targets = [taylor_green_solution(coords, nu=nu).reshape(ny, nx, 3).permute(2, 0, 1) for coords in coords_list]
    return torch.stack(targets, dim=0)


def train_fno(args: argparse.Namespace) -> dict[str, float]:
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = device_from_arg(args.device)
    domain = Domain()
    result_dir = Path(args.output_dir) / args.run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    model = make_model("fno", width=args.width, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, float]] = []
    start = time.perf_counter()
    peak_mem = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, args.steps + 1):
        t_batch = args.train_t_max * torch.rand(args.fno_batch, device=device)
        x = fno_input_grid(t_batch, args.fno_grid, args.fno_grid, domain, device)
        y = fno_target_grid(t_batch, args.fno_grid, args.fno_grid, domain, device, args.nu)
        if args.noise_std > 0:
            y = add_noise(y.flatten(2).transpose(1, 2), args.noise_std, args.seed + step).transpose(1, 2).reshape_as(y)
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            if device.type == "cuda":
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(device))
            history.append({"step": step, "loss": float(loss.detach().cpu()), "data_loss": float(loss.detach().cpu()), "pde_loss": 0.0, "continuity_loss": 0.0})

    train_seconds = time.perf_counter() - start
    preds = []
    trues = []
    for t_value in [0.25, 0.50, 0.75]:
        t_tensor = torch.tensor([t_value], device=device)
        x = fno_input_grid(t_tensor, 72, 72, domain, device)
        with torch.no_grad():
            pred = model(x)[0].permute(1, 2, 0).reshape(-1, 3)
            true = fno_target_grid(t_tensor, 72, 72, domain, device, args.nu)[0].permute(1, 2, 0).reshape(-1, 3)
        preds.append(pred)
        trues.append(true)
    pred_all = torch.cat(preds, dim=0)
    true_all = torch.cat(trues, dim=0)
    metrics = channel_metrics(pred_all, true_all)
    metrics.update(
        {
            "model": "fno",
            "run_name": args.run_name,
            "n_obs": args.n_obs,
            "noise_std": args.noise_std,
            "train_t_max": args.train_t_max,
            "steps": args.steps,
            "train_seconds": train_seconds,
            "peak_gpu_mb": peak_mem / (1024**2),
            "lambda_1": None,
            "lambda_2": None,
            "lambda_1_abs_error": None,
            "lambda_2_abs_error": None,
            "pde_residual_mse": None,
            "continuity_mse": None,
        }
    )
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    write_history(history, result_dir / "history.csv")
    plot_history(history, result_dir / "loss_curve.png")
    torch.save({"model_state": model.state_dict(), "metrics": metrics}, result_dir / "checkpoint.pt")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ECE 228 PINN project models.")
    parser.add_argument("--model", choices=["fcn", "deep_fcn", "fourier", "siren", "residual", "fno"], default="fcn")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--n-obs", type=int, default=4096)
    parser.add_argument("--data-batch", type=int, default=1024)
    parser.add_argument("--colloc-batch", type=int, default=1024)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--train-t-max", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--lambda1-init", type=float, default=0.5)
    parser.add_argument("--lambda2-init", type=float, default=0.02)
    parser.add_argument("--p-data-weight", type=float, default=1.0)
    parser.add_argument("--pde-weight", type=float, default=1.0)
    parser.add_argument("--continuity-weight", type=float, default=1.0)
    parser.add_argument("--pde-warmup-steps", type=int, default=200)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--fno-grid", type=int, default=48)
    parser.add_argument("--fno-batch", type=int, default=4)
    args = parser.parse_args()
    if args.run_name is None:
        noise = str(args.noise_std).replace(".", "p")
        args.run_name = f"{args.model}_n{args.n_obs}_noise{noise}_s{args.seed}"
    return args


def main() -> None:
    args = parse_args()
    metrics = train_fno(args) if args.model == "fno" else train_coord_model(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
