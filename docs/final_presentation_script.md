# Final Presentation Script

## Slide 1: Motivation

Our project started with a simple idea: use a PINN to learn a Navier-Stokes fluid field. But after the proposal feedback, we realized that just training one fully connected network would be too close to a basic regression task. So we changed the project into a model comparison: which architecture actually works best for learning a physical vector field?

## Slide 2: Task

We use the 2D Taylor-Green vortex, which is an analytic solution to incompressible Navier-Stokes. The input is `(x, y, t)`, and the model predicts velocity `u`, velocity `v`, and pressure `p`. For PINN models, we also learn the physical coefficients `lambda_1` and `lambda_2`, and we penalize both the Navier-Stokes residual and the continuity residual.

## Slide 3: Methods

We compare six models. The first is a vanilla FCN PINN. Then we test a deeper FCN, Fourier-feature PINN, SIREN PINN, residual PINN, and an FNO baseline. This lets us compare standard coordinate learning, specialized coordinate networks, and a neural-operator-style model.

## Slide 4: Experiments

We do more than hyperparameter tuning. We run architecture comparison, data scarcity, noise robustness, time extrapolation, FNO resolution, and ablation studies. We evaluate relative L2 error, PDE residual, continuity residual, physical coefficient recovery, training time, and GPU memory.

## Slide 5: Main Result

The strongest full inverse-PINN model is the residual PINN. It has the best velocity error among PINN-style models and also recovers `lambda_1` and `lambda_2` most accurately. FNO is also very strong and fast for field prediction, but in our setup it does not recover the physical coefficients.

## Slide 6: Takeaway

The main lesson is that good field prediction and good physics recovery are not exactly the same thing. Fourier and SIREN can fit the field reasonably well, but their PDE residual or coefficient recovery can be worse. Residual PINN gives the best balance between prediction accuracy and physical consistency.

## Q&A Notes

If asked why we used Taylor-Green vortex:

We wanted a controlled benchmark with exact ground truth and exact physical coefficients, so we could run many architecture comparisons reproducibly. The original cylinder wake example is still a natural next step.

If asked why FNO has no lambda error:

Our current FNO is a neural-operator field predictor. It predicts the whole field on a grid, but it is not trained as an inverse PINN with learnable physical coefficients.

If asked what we would improve:

We would add physics-informed FNO, DeepONet, adaptive collocation, multi-seed runs, and then transfer the comparison to cylinder wake data.

