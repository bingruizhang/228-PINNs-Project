$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

conda run -n 5080py python -m pinn_project.train --model fcn --run-name smoke_fcn --steps 20 --n-obs 512 --data-batch 128 --colloc-batch 128 --width 32 --depth 3 --log-every 10
conda run -n 5080py python -m pinn_project.train --model fourier --run-name smoke_fourier --steps 20 --n-obs 512 --data-batch 128 --colloc-batch 128 --width 32 --depth 3 --log-every 10
conda run -n 5080py python -m pinn_project.train --model siren --run-name smoke_siren --steps 20 --n-obs 512 --data-batch 128 --colloc-batch 128 --width 32 --depth 3 --log-every 10
conda run -n 5080py python -m pinn_project.train --model residual --run-name smoke_residual --steps 20 --n-obs 512 --data-batch 128 --colloc-batch 128 --width 32 --depth 3 --log-every 10
conda run -n 5080py python -m pinn_project.train --model fno --run-name smoke_fno --steps 20 --width 32 --fno-grid 24 --fno-batch 2 --log-every 10
conda run -n 5080py python -m pinn_project.summarize --results-dir results --out results/smoke_summary.csv

