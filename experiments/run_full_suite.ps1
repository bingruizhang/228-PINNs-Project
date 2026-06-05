$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

# Architecture comparison.
foreach ($model in @("fcn", "deep_fcn", "fourier", "residual", "fno")) {
  conda run -n 5080py python -m pinn_project.train --model $model --run-name "arch_$model" --steps 4000 --n-obs 5000 --data-batch 2048 --colloc-batch 2048 --width 128 --depth 6 --log-every 250
}
conda run -n 5080py python -m pinn_project.train --model siren --run-name "arch_siren" --steps 4000 --n-obs 5000 --data-batch 2048 --colloc-batch 2048 --width 128 --depth 3 --log-every 250

# Data-scarcity study.
foreach ($n in @(500, 1000, 2500, 5000, 10000)) {
  foreach ($model in @("fcn", "fourier", "residual")) {
    conda run -n 5080py python -m pinn_project.train --model $model --run-name "data_${model}_n$n" --steps 3000 --n-obs $n --data-batch 1024 --colloc-batch 2048 --width 128 --depth 6 --log-every 250
  }
  conda run -n 5080py python -m pinn_project.train --model siren --run-name "data_siren_n$n" --steps 3000 --n-obs $n --data-batch 1024 --colloc-batch 2048 --width 128 --depth 3 --log-every 250
}

# Noise robustness study.
foreach ($noise in @(0.0, 0.01, 0.05, 0.10)) {
  foreach ($model in @("fcn", "fourier", "residual")) {
    $tag = $noise.ToString().Replace(".", "p")
    conda run -n 5080py python -m pinn_project.train --model $model --run-name "noise_${model}_$tag" --steps 3000 --n-obs 5000 --noise-std $noise --data-batch 2048 --colloc-batch 2048 --width 128 --depth 6 --log-every 250
  }
  $tag = $noise.ToString().Replace(".", "p")
  conda run -n 5080py python -m pinn_project.train --model siren --run-name "noise_siren_$tag" --steps 3000 --n-obs 5000 --noise-std $noise --data-batch 2048 --colloc-batch 2048 --width 128 --depth 3 --log-every 250
}

# Time extrapolation study: train only on early time window, evaluate on t=0.25, 0.50, 0.75.
foreach ($tmax in @(0.35, 0.50, 0.75, 1.00)) {
  foreach ($model in @("fcn", "fourier", "residual")) {
    $tag = $tmax.ToString().Replace(".", "p")
    conda run -n 5080py python -m pinn_project.train --model $model --run-name "time_${model}_$tag" --steps 3000 --n-obs 5000 --train-t-max $tmax --data-batch 2048 --colloc-batch 2048 --width 128 --depth 6 --log-every 250
  }
  $tag = $tmax.ToString().Replace(".", "p")
  conda run -n 5080py python -m pinn_project.train --model siren --run-name "time_siren_$tag" --steps 3000 --n-obs 5000 --train-t-max $tmax --data-batch 2048 --colloc-batch 2048 --width 128 --depth 3 --log-every 250
}

# Neural operator resolution study.
foreach ($grid in @(24, 32, 48, 64)) {
  conda run -n 5080py python -m pinn_project.train --model fno --run-name "resolution_fno_g$grid" --steps 3000 --width 128 --fno-grid $grid --fno-batch 4 --log-every 250
}

# Fourier/SIREN representation ablations.
foreach ($depth in @(3, 5, 8)) {
  conda run -n 5080py python -m pinn_project.train --model fourier --run-name "ablate_fourier_depth$depth" --steps 3000 --n-obs 5000 --width 128 --depth $depth --log-every 250
  conda run -n 5080py python -m pinn_project.train --model siren --run-name "ablate_siren_depth$depth" --steps 3000 --n-obs 5000 --width 128 --depth $depth --log-every 250
}

conda run -n 5080py python -m pinn_project.summarize --results-dir results --out results/final_summary.csv --exclude-prefix smoke_
conda run -n 5080py python -m pinn_project.make_figures --summary results/final_summary.csv --out-dir results/figures
