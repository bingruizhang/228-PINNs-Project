$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

# A medium-size run that produces useful preliminary figures in much less time than the full suite.

foreach ($model in @("fcn", "deep_fcn", "fourier", "siren", "residual", "fno")) {
  conda run -n 5080py python -m pinn_project.train --model $model --run-name "arch_$model" --steps 800 --n-obs 3000 --data-batch 1024 --colloc-batch 1024 --width 64 --depth 5 --fno-grid 32 --fno-batch 3 --log-every 100
}

foreach ($n in @(500, 2500, 5000)) {
  foreach ($model in @("fcn", "fourier", "siren", "residual")) {
    conda run -n 5080py python -m pinn_project.train --model $model --run-name "data_${model}_n$n" --steps 500 --n-obs $n --data-batch 1024 --colloc-batch 1024 --width 64 --depth 5 --log-every 100
  }
}

foreach ($noise in @(0.0, 0.05, 0.10)) {
  foreach ($model in @("fcn", "fourier", "siren", "residual")) {
    $tag = $noise.ToString().Replace(".", "p")
    conda run -n 5080py python -m pinn_project.train --model $model --run-name "noise_${model}_$tag" --steps 500 --n-obs 3000 --noise-std $noise --data-batch 1024 --colloc-batch 1024 --width 64 --depth 5 --log-every 100
  }
}

foreach ($tmax in @(0.50, 0.75, 1.00)) {
  foreach ($model in @("fcn", "fourier", "siren", "residual")) {
    $tag = $tmax.ToString().Replace(".", "p")
    conda run -n 5080py python -m pinn_project.train --model $model --run-name "time_${model}_$tag" --steps 500 --n-obs 3000 --train-t-max $tmax --data-batch 1024 --colloc-batch 1024 --width 64 --depth 5 --log-every 100
  }
}

foreach ($grid in @(24, 48)) {
  conda run -n 5080py python -m pinn_project.train --model fno --run-name "resolution_fno_g$grid" --steps 500 --width 64 --fno-grid $grid --fno-batch 3 --log-every 100
}

conda run -n 5080py python -m pinn_project.summarize --results-dir results --out results/final_summary.csv --exclude-prefix smoke_
conda run -n 5080py python -m pinn_project.make_figures --summary results/final_summary.csv --out-dir results/figures
