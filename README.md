# ECE 228 Final Project: PINNs and Neural Operators for Inverse Navier-Stokes Learning

## Project idea

Our original proposal used a fully connected PINN to solve a Navier-Stokes example. The instructor feedback was that this was too close to a basic regression task unless we made the project more technically compelling. We therefore upgraded the project into a comparative study:

> How do vanilla PINNs, specialized coordinate networks, and neural-operator-style models compare on inverse Navier-Stokes field learning?

The main controlled task uses the analytic 2D Taylor-Green vortex solution of the incompressible Navier-Stokes equations. This gives us exact velocity and pressure fields, exact PDE residuals, and known physical parameters. We can evaluate not only prediction error, but also physical consistency and parameter recovery.

## Models included

- `fcn`: vanilla fully connected PINN baseline.
- `deep_fcn`: deeper FCN ablation.
- `fourier`: Fourier-feature PINN for coordinate-conditioned prediction.
- `siren`: sinusoidal representation network PINN.
- `residual`: residual MLP PINN.
- `fno`: small Fourier Neural Operator style grid model.

## Experiments included

- Architecture comparison.
- Data scarcity study with different numbers of observed points.
- Noise robustness study.
- Time extrapolation study.
- FNO resolution study.
- Fourier/SIREN representation ablations.
- Runtime and GPU-memory tracking.

## Environment

Use the local `5080py` conda environment:

```powershell
conda run -n 5080py python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

For all project commands, run from the repository root and set `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "src"
```

## Quick smoke test

This verifies that every implemented model trains, evaluates, and writes outputs:

```powershell
powershell -ExecutionPolicy Bypass -File experiments\run_smoke.ps1
```

Outputs:

- `results/smoke_*/metrics.json`
- `results/smoke_*/history.csv`
- `results/smoke_*/loss_curve.png`
- `results/smoke_*/slice_t050.png` for coordinate PINNs
- `results/smoke_summary.csv`

## Full experiment suite

This is the intended final-project run. It is larger and should be run on the RTX 5080:

```powershell
powershell -ExecutionPolicy Bypass -File experiments\run_full_suite.ps1
```

For a faster but still meaningful run before the final overnight suite:

```powershell
powershell -ExecutionPolicy Bypass -File experiments\run_preliminary_suite.ps1
```

Outputs:

- `results/final_summary.csv`
- `results/figures/architecture_summary.png`
- `results/figures/data_scarcity.png`
- `results/figures/noise_robustness.png`
- `results/figures/time_generalization.png`
- per-run checkpoints, metrics, histories, and prediction plots

## Run one experiment manually

```powershell
$env:PYTHONPATH = "src"
conda run -n 5080py python -m pinn_project.train --model fourier --run-name demo_fourier --steps 3000 --n-obs 5000 --data-batch 2048 --colloc-batch 2048 --width 128 --depth 6
```

Summarize all results:

```powershell
conda run -n 5080py python -m pinn_project.summarize --results-dir results --out results/final_summary.csv
conda run -n 5080py python -m pinn_project.make_figures --summary results/final_summary.csv --out-dir results/figures
```

## Repository structure

- `src/pinn_project/`: self-contained PyTorch final-project code.
- `src/examples/`: original `pinns-torch` reference examples kept from the proposal stage.
- `experiments/`: smoke and full experiment PowerShell scripts.
- `results/`: generated metrics, plots, and checkpoints.
- `docs/`: proposal, handoff notes, and final report/presentation planning.

## Main report message

The report should not say that we only tuned hyperparameters. The stronger story is:

1. Vanilla FCN PINNs are a necessary baseline, but their coordinate representation can be weak.
2. Fourier features and SIREN directly target the coordinate-conditioned nature of PDE field prediction.
3. FNO represents the neural-operator direction from Week 7 and gives a grid/function-to-function comparison.
4. We evaluate prediction accuracy, physical residual, parameter recovery, robustness, data efficiency, and computational cost.
