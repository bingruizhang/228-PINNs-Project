# Comparing PINNs, Coordinate Networks, and Neural Operators for Inverse Navier-Stokes Learning

**Team:** Bingrui Zhang, Yijia Dou, Haifan Zhao, Zhexi Feng

## Abstract

Physics-informed neural networks (PINNs) provide a way to train neural networks with both data supervision and physical equation constraints. Our original project idea was to train a fully connected PINN on a Navier-Stokes example, but this alone is close to a standard coordinate-regression task. In this project, we instead compare several model families for learning a continuous fluid vector field: vanilla FCN PINNs, deeper FCN PINNs, Fourier-feature PINNs, SIREN PINNs, residual PINNs, and a small Fourier Neural Operator (FNO) baseline. We use the 2D Taylor-Green vortex as a controlled Navier-Stokes benchmark because it provides exact velocity, pressure, and physical coefficients. We evaluate relative field error, PDE residual, continuity error, inverse coefficient recovery, robustness to noise, data efficiency, time extrapolation, and computational cost. Our main result is that residual PINNs give the strongest full inverse-PINN performance, while FNO is a fast and accurate field-prediction baseline. We also find that expressive coordinate networks such as Fourier features and SIREN can improve field fitting, but do not automatically improve physical parameter recovery.

## 1. Introduction

Learning physical systems with neural networks is useful when direct simulation or repeated numerical solving is expensive. However, a standard neural network trained only with data loss may produce predictions that look accurate but violate the governing physical equations. This issue is especially important for fluid dynamics, where velocity and pressure fields must satisfy conservation laws and the Navier-Stokes equations.

Our starting point was a Physics-Informed Neural Network (PINN) for Navier-Stokes. A PINN adds the PDE residual to the loss function, so the model is encouraged to match data and obey physics at the same time. After receiving instructor feedback, we realized that only training a fully connected network and tuning its hyperparameters would be too basic for a final project. We therefore changed the project into a broader architecture comparison.

Our project asks:

> Which model architectures are most effective for inverse Navier-Stokes field learning, and do better field predictions also imply better physical consistency?

We compare pointwise coordinate-based PINNs with specialized coordinate representations and a neural-operator-style grid model. This connects directly to the course topics: feedforward networks, coordinate representations, neural operators, and PINNs.

Our contributions are:

1. We implement a multi-architecture benchmark for inverse Navier-Stokes learning.
2. We compare vanilla FCN PINNs with Fourier-feature, SIREN, and residual PINN variants.
3. We include a Fourier Neural Operator baseline to compare coordinate learning with grid/function learning.
4. We evaluate not only field error, but also PDE residual, continuity error, physical parameter recovery, robustness, extrapolation, and efficiency.

## 2. Related Work

### Physics-Informed Neural Networks

PINNs were introduced as a way to solve forward and inverse problems involving nonlinear PDEs by adding physics residuals to the neural network loss. Instead of requiring a dense labeled dataset everywhere in the domain, PINNs can use collocation points where the PDE is enforced. This makes them attractive for physical systems where data may be limited or expensive.

However, PINNs are also known to be difficult to train. The data loss and PDE loss can compete, derivatives computed by autograd may become unstable, and the network may satisfy the residual while still failing to represent the correct solution. Because of this, architecture and training design matter.

### Coordinate-Based Neural Fields

A fluid solution can be viewed as a continuous field: for each coordinate `(x, y, t)`, the model predicts `u`, `v`, and `p`. A vanilla MLP can represent such a mapping, but coordinate networks can suffer from spectral bias and may learn low-frequency components more easily than high-frequency details.

Fourier features address this by mapping input coordinates into sinusoidal features before the MLP. SIREN goes further by using sinusoidal activations throughout the network. These methods are designed for continuous signal representation, so they are natural candidates for PDE field learning.

### Neural Operators

Neural operators, such as DeepONet and FNO, learn mappings between functions rather than only fixed-dimensional vectors. FNO uses Fourier-domain layers to model global spatial interactions efficiently. This is different from a pointwise PINN because it predicts a whole field on a grid. In our project, FNO serves as a neural-operator-style baseline for field prediction.

## 3. Problem Setup

We use the 2D incompressible Navier-Stokes equations:

```text
u_t + lambda_1 (u u_x + v u_y) = -p_x + lambda_2 (u_xx + u_yy)
v_t + lambda_1 (u v_x + v v_y) = -p_y + lambda_2 (v_xx + v_yy)
u_x + v_y = 0
```

The model receives coordinates `(x, y, t)` and predicts the velocity and pressure field:

```text
f_theta(x, y, t) -> [u, v, p]
```

For the controlled benchmark, we use the Taylor-Green vortex analytic solution:

```text
u = -cos(x) sin(y) exp(-2 nu t)
v =  sin(x) cos(y) exp(-2 nu t)
p = -0.25 [cos(2x) + cos(2y)] exp(-4 nu t)
```

with `lambda_1 = 1` and `lambda_2 = nu = 0.01`.

This benchmark is useful because it gives exact ground truth for field values and physical coefficients. It also lets us run many experiments reproducibly without relying on a large external dataset.

## 4. Method

### PINN Loss

For coordinate-based models, we optimize:

```text
L = L_data + w_pde L_pde + w_cont L_cont
```

where `L_data` supervises `u`, `v`, and `p`, `L_pde` penalizes the two Navier-Stokes momentum residuals, and `L_cont` penalizes the incompressibility residual `u_x + v_y`.

The physical coefficients `lambda_1` and `lambda_2` are learned jointly with the network. We parameterize them with a positive transform to avoid negative viscosity.

### Models

**FCN PINN:** A vanilla fully connected network. This is the baseline closest to the original proposal.

**Deep FCN PINN:** A deeper version of the FCN baseline. This checks whether simply adding depth improves performance.

**Fourier-feature PINN:** A coordinate MLP with sinusoidal input features. This tests whether better coordinate encoding improves field learning.

**SIREN PINN:** A sinusoidal representation network. We found that a shallow SIREN is much more stable than a deep SIREN for this task.

**Residual PINN:** A residual MLP with skip connections. This improves optimization while keeping the PINN formulation.

**FNO:** A small Fourier Neural Operator style model. It predicts whole fields on a grid and serves as a neural-operator baseline. In our current implementation, it does not estimate `lambda_1` and `lambda_2`, so we compare it only as a field predictor.

## 5. Experiments

We ran the full experiment suite on an RTX 5080 using the `5080py` environment. Results are saved in `results/final_summary.csv`.

### Metrics

We report:

- relative L2 error for `u`, `v`, and `p`
- absolute error for `lambda_1` and `lambda_2`
- PDE residual MSE
- continuity residual MSE
- training time
- peak GPU memory

### Experiment Groups

**Architecture comparison:** all models trained with the same main setting.

**Data scarcity:** `n_obs = 500, 1000, 2500, 5000, 10000`.

**Noise robustness:** observation noise of `0%`, `1%`, `5%`, and `10%`.

**Time extrapolation:** train only up to `t_max = 0.35, 0.50, 0.75, 1.00` and evaluate on common later slices.

**FNO resolution:** train FNO with grids `24, 32, 48, 64`.

**Representation ablation:** compare Fourier and SIREN depths.

## 6. Results

The main architecture comparison is:

| Model | rel L2 u | rel L2 v | rel L2 p | lambda1 abs err | lambda2 abs err |
|---|---:|---:|---:|---:|---:|
| FCN PINN | 0.0422 | 0.0574 | 0.0651 | 0.0098 | 0.0013 |
| Deep FCN PINN | 0.0268 | 0.0188 | 0.0750 | 0.0145 | 0.0014 |
| Fourier PINN | 0.0576 | 0.0569 | 0.0749 | 0.6301 | 0.0086 |
| SIREN PINN | 0.0176 | 0.0194 | 0.0348 | 0.3556 | 0.0048 |
| Residual PINN | 0.0110 | 0.0084 | 0.0301 | 0.0031 | 0.0003 |
| FNO | 0.0200 | 0.0204 | 0.0314 | N/A | N/A |

The residual PINN is the strongest full inverse-PINN model. It has the lowest coordinate-model velocity error and the best recovery of `lambda_1` and `lambda_2`. This suggests that skip connections help optimization while preserving physical consistency.

FNO is very strong for field prediction and is much faster than most coordinate PINNs. However, since our FNO is not trained with PDE residuals and does not recover physical coefficients, it should be interpreted as a strong neural-operator field-prediction baseline rather than a complete inverse PINN.

Fourier features and SIREN show an important nuance. They can produce good field predictions, especially after tuning SIREN depth, but their physical parameter recovery is weaker than residual PINN. This supports the idea that supervised field error and physical consistency are not the same metric.

## 7. Discussion

The instructor feedback helped us move from a basic regression project to a more meaningful architecture study. The results show that architecture choice matters a lot for PDE learning.

One interesting result is that the best field predictor is not always the best physical inverse model. For example, FNO is accurate and fast, but it does not recover physical parameters in our setup. Fourier and SIREN also show that expressive representations may fit fields while still having higher PDE residuals or weaker coefficient recovery.

The residual PINN gives the best balance. It is still a PINN, so it can use PDE residuals and learn physical coefficients, but the skip connections make optimization easier than a plain FCN.

## 8. Limitations and Future Work

Our benchmark uses an analytic Taylor-Green vortex rather than the full cylinder-wake dataset from the original `pinns-torch` example. This was a deliberate choice to make the architecture comparison controlled and reproducible. A next step would be to run the same model comparison on the cylinder wake data.

Other future directions include:

- adding DeepONet as a second neural-operator baseline
- training FNO with physics-informed residuals
- adaptive collocation sampling
- better loss balancing for Fourier and SIREN PINNs
- multi-seed statistical analysis
- geometry-informed FNO for more realistic domains

## 9. Conclusion

We compared vanilla PINNs, specialized coordinate networks, residual PINNs, and FNO for inverse Navier-Stokes learning. The main finding is that residual PINNs provide the strongest physics-informed performance, while FNO is a strong fast field-prediction baseline. The project also shows that field accuracy alone is not enough for physical ML; PDE residuals and parameter recovery reveal important differences between models.

