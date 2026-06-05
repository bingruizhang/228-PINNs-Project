# ECE 228 Final Project Package Contents

This package is organized so teammates can open it directly after downloading from Google Drive.

## Start here

1. `README.md` - short project overview and reproduction commands.
2. `docs/project_handoff.md` - full detailed handoff document in Chinese.
3. `docs/final_results_summary.md` - main numerical results and takeaways.
4. `docs/final_report_draft.md` - draft final report text.
5. `docs/final_presentation_script.md` - draft 3-minute presentation script.

## Main folders

- `src/pinn_project/` - self-contained PyTorch implementation for the final project.
- `experiments/` - PowerShell scripts for smoke tests, preliminary runs, full runs, and optimized SIREN reruns.
- `results/` - completed experiment outputs, including metrics, histories, plots, checkpoints, and summary figures.
- `docs/` - proposal, final report planning, final report draft, presentation plan, handoff, and results summary.
- `src/examples/` - original `pinns-torch` reference examples kept from the proposal stage.
- `notebooks/` - original tutorial notebook kept as reference.
- `data/` - generated benchmark data notes and a small sample `.npz`; the main training data is generated analytically by code.

## Most important result files

- `results/final_summary.csv`
- `results/figures/architecture_summary.png`
- `results/figures/data_scarcity.png`
- `results/figures/noise_robustness.png`
- `results/figures/time_generalization.png`
- `results/arch_residual/slice_t050.png`
- `results/arch_fcn/slice_t050.png`
- `results/arch_siren/slice_t050.png`

## Data note

The final experiments do not need a large external dataset. The Taylor-Green vortex benchmark is generated analytically in `src/pinn_project/data.py`. A small optional inspection file is included at `data/taylor_green_sample.npz`.

## Reproduction environment

Use the local conda environment:

```powershell
conda run -n 5080py python ...
```

Before running Python modules:

```powershell
$env:PYTHONPATH = "src"
```

Quick check:

```powershell
powershell -ExecutionPolicy Bypass -File experiments\run_smoke.ps1
```

Full suite:

```powershell
powershell -ExecutionPolicy Bypass -File experiments\run_full_suite.ps1
```
