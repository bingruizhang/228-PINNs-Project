$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

# Re-run all SIREN entries with the stable shallow SIREN setting found in the ablation study.

conda run -n 5080py python -m pinn_project.train --model siren --run-name "arch_siren" --steps 4000 --n-obs 5000 --data-batch 2048 --colloc-batch 2048 --width 128 --depth 3 --log-every 250

foreach ($n in @(500, 1000, 2500, 5000, 10000)) {
  conda run -n 5080py python -m pinn_project.train --model siren --run-name "data_siren_n$n" --steps 3000 --n-obs $n --data-batch 1024 --colloc-batch 2048 --width 128 --depth 3 --log-every 250
}

foreach ($noise in @(0.0, 0.01, 0.05, 0.10)) {
  $tag = $noise.ToString().Replace(".", "p")
  conda run -n 5080py python -m pinn_project.train --model siren --run-name "noise_siren_$tag" --steps 3000 --n-obs 5000 --noise-std $noise --data-batch 2048 --colloc-batch 2048 --width 128 --depth 3 --log-every 250
}

foreach ($tmax in @(0.35, 0.50, 0.75, 1.00)) {
  $tag = $tmax.ToString().Replace(".", "p")
  conda run -n 5080py python -m pinn_project.train --model siren --run-name "time_siren_$tag" --steps 3000 --n-obs 5000 --train-t-max $tmax --data-batch 2048 --colloc-batch 2048 --width 128 --depth 3 --log-every 250
}

conda run -n 5080py python -m pinn_project.summarize --results-dir results --out results/final_summary.csv --exclude-prefix smoke_
conda run -n 5080py python -m pinn_project.make_figures --summary results/final_summary.csv --out-dir results/figures

