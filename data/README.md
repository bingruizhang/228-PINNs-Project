# Data Folder

The final project does not require a large external dataset.

Our controlled benchmark uses the analytic 2D Taylor-Green vortex solution of the incompressible Navier-Stokes equations. Training and collocation points are generated on the fly by:

```text
src/pinn_project/data.py
```

This is why the project package is relatively small. The dataset is defined mathematically rather than stored as a large file.

For convenience, this folder also contains a small pre-generated sample file:

```text
taylor_green_sample.npz
```

It is not required for training. It is included only so teammates can quickly inspect the benchmark arrays without running the full training code.

The original `pinns-torch` cylinder wake dataset is not included in this package because our final experiments use the analytic Taylor-Green benchmark for controlled, reproducible architecture comparison.

