# Final Report Plan

## Title

Comparing PINNs, Coordinate Networks, and Neural Operators for Inverse Navier-Stokes Learning

## 1. Introduction

Goal: explain why predicting fluid fields is useful and why ordinary neural networks may violate physics.

Main message:

- A vanilla FCN PINN is a reasonable starting point, but the instructor feedback correctly pointed out that it is too basic alone.
- The project studies whether specialized representations help when the output is a continuous physical vector field.
- We evaluate both data accuracy and physical consistency.

Contributions:

1. Multi-architecture benchmark for inverse Navier-Stokes learning.
2. Coordinate-network variants: Fourier PINN and SIREN PINN.
3. Neural-operator-style grid baseline: FNO.
4. Evaluation across accuracy, physical residual, parameter recovery, noise, data scarcity, extrapolation, and efficiency.

## 2. Related Work

Organize by themes:

- Physics-informed neural networks: add PDE residuals to the loss.
- Coordinate-based neural fields: Fourier features and SIREN improve continuous signal representation.
- Neural operators: DeepONet and FNO learn mappings between functions rather than only fixed vectors.
- PINN training difficulties: optimization, loss balancing, spectral bias, noisy data.

## 3. Method

Task:

- Incompressible 2D Navier-Stokes equation.
- Taylor-Green vortex analytic solution.
- Predict `u`, `v`, `p`; recover `lambda_1`, `lambda_2`.

Loss:

- Data loss on observed fields.
- PDE momentum residual loss.
- Continuity residual loss.
- Optional pressure supervision weight.

Models:

- FCN PINN.
- Deep FCN PINN.
- Fourier-feature PINN.
- SIREN PINN.
- Residual PINN.
- FNO grid baseline.

## 4. Experiments

Architecture comparison:

- Compare all models on the same data budget.

Data scarcity:

- `n_obs = 500, 1000, 2500, 5000, 10000`.

Noise robustness:

- noise levels `0, 1%, 5%, 10%`.

Time extrapolation:

- train with `t_max = 0.35, 0.50, 0.75, 1.00`; evaluate on later slices.

Resolution study:

- FNO train grids `24, 32, 48, 64`; evaluate on `72`.

Efficiency:

- training seconds, inference behavior, GPU memory.

## 5. Results and Analysis

Figures to include:

- Prediction/error heatmaps for `u`, `v`, `p`.
- Architecture summary bar chart.
- Data scarcity curve.
- Noise robustness curve.
- Time generalization curve.
- Training time vs error scatter.
- Lambda recovery table.

Analysis questions:

- Does Fourier/SIREN improve field reconstruction?
- Does better supervised error also mean lower PDE residual?
- Which models recover physical parameters best?
- Which model is best under sparse/noisy data?
- What is the cost of each model?

Current result narrative:

- Residual PINN is the strongest full inverse-PINN model because it performs well on field prediction, PDE residual, and `lambda` recovery.
- FNO is the fastest strong field predictor, but it should be described as an operator-style baseline, not a coefficient-recovery model.
- Fourier features and SIREN show that expressive coordinate networks are not automatically better for physics consistency. This creates a useful discussion about representation versus physical parameter recovery.

## 6. Conclusion

Summarize:

- Vanilla PINN is useful but not enough as the whole project.
- Specialized coordinate representations are better suited for continuous physical fields.
- Neural-operator-style models provide a useful comparison for grid/function prediction.
- Future work: cylinder wake data, DeepONet, geometry-informed FNO, adaptive collocation, better PINN loss balancing.
