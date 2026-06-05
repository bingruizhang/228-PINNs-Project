# 3-Minute Presentation Plan

## Slide 1: Problem and Motivation

Message:

We started from a simple PINN idea, but the instructor feedback pushed us to ask a better question: which architecture is actually suitable for learning a physical vector field?

Visual:

- One small fluid-field image.
- One sentence: "Predict velocity/pressure while respecting Navier-Stokes."

## Slide 2: Method

Message:

We compare three families:

- Vanilla PINNs.
- Better coordinate networks: Fourier features and SIREN.
- Neural-operator-style FNO.

Visual:

- Simple model comparison diagram.
- Loss equation: data loss + PDE residual + continuity loss.

## Slide 3: Experiments

Message:

We do more than tune hyperparameters.

Visual:

- Compact table with experiment types:
  - architecture
  - data scarcity
  - noise
  - time extrapolation
  - resolution
  - efficiency

## Slide 4: Main Result

Message:

Show the strongest result from the completed full suite.

Visual options:

- Architecture summary figure.
- One heatmap triplet: truth / prediction / error.
- Accuracy-time Pareto plot.

Suggested line:

Residual PINN gives the best physics-informed result, while FNO is a very fast field predictor. The interesting lesson is that good field prediction and good physics recovery are related but not identical.

## Slide 5: Takeaway

Message:

The main insight should be specific, for example:

> Better coordinate representations reduce field error and help training stability, while FNO gives a useful whole-grid prediction baseline.

End with limitation:

- Controlled analytic benchmark first.
- Future work: cylinder wake and geometry-informed neural operators.
