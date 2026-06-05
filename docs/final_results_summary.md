# Final Experiment Summary

## What has been run

The full experiment suite has been completed on the local RTX 5080 using the `5080py` environment. The final summary contains 68 experiment rows:

- architecture comparison
- data scarcity
- noise robustness
- time extrapolation
- FNO resolution study
- Fourier/SIREN ablation

Main files:

- `results/final_summary.csv`
- `results/figures/architecture_summary.png`
- `results/figures/data_scarcity.png`
- `results/figures/noise_robustness.png`
- `results/figures/time_generalization.png`

## Architecture comparison

| Model | rel L2 u | rel L2 v | rel L2 p | lambda1 abs err | lambda2 abs err | Notes |
|---|---:|---:|---:|---:|---:|---|
| FCN PINN | 0.0422 | 0.0574 | 0.0651 | 0.0098 | 0.0013 | Strong baseline after enough training. |
| Deep FCN PINN | 0.0268 | 0.0188 | 0.0750 | 0.0145 | 0.0014 | Better velocity field, slightly worse parameter recovery. |
| Fourier PINN | 0.0576 | 0.0569 | 0.0749 | 0.6301 | 0.0086 | Fits fields moderately but weak physical parameter recovery. |
| SIREN PINN | 0.0176 | 0.0194 | 0.0348 | 0.3556 | 0.0048 | Good field fit after shallow-depth tuning; parameter recovery still weak. |
| Residual PINN | 0.0110 | 0.0084 | 0.0301 | 0.0031 | 0.0003 | Best PINN-style model overall. |
| FNO | 0.0200 | 0.0204 | 0.0314 | N/A | N/A | Very strong and fast field predictor; not an inverse parameter model. |

## Main takeaways

1. Residual PINN is the strongest physics-informed coordinate model. It gives the lowest velocity error among PINN variants and also recovers the physical coefficients most accurately.

2. FNO is highly competitive for full-field prediction and is much faster. However, in this implementation it is not trained with PDE residuals and does not estimate `lambda_1` or `lambda_2`, so it should be presented as a neural-operator field-prediction baseline rather than a full inverse PINN.

3. Fourier features improve coordinate expressivity but can weaken physical consistency. This is visible in its acceptable field error but poor `lambda_1` recovery and high PDE residual. This is a useful result: lower supervised error does not automatically mean better physics.

4. SIREN is sensitive to architecture depth. The deep SIREN collapsed during the first full run, while the shallow SIREN found through ablation produced good field error. This supports a discussion about optimization difficulty in sinusoidal networks.

5. The project now directly addresses the instructor feedback. It is no longer only FCN regression or hyperparameter tuning; it compares model families, studies robustness/generalization, and evaluates physical consistency.

## Best figures for report and slides

Use these first:

- `results/figures/architecture_summary.png`
- `results/figures/data_scarcity.png`
- `results/figures/noise_robustness.png`
- `results/arch_residual/slice_t050.png`
- `results/arch_fcn/slice_t050.png`
- `results/arch_siren/slice_t050.png`

For the 3-minute presentation, the best story is:

1. Start with the instructor's concern: vanilla FCN PINN alone is too basic.
2. Show the architecture comparison.
3. Emphasize the key insight: residual connections helped both field prediction and physical parameter recovery, while FNO was strong for fast full-field prediction.
4. End with the nuance: field accuracy and physical consistency are not always the same thing.

## Caution for writing

Do not claim that FNO solves the full inverse problem in the same way as PINN. In our current code, FNO predicts fields from grid/time inputs but does not use PDE residuals or recover coefficients.

Do not hide the Fourier/SIREN weaknesses. They are useful analysis points and make the report more honest and interesting.

