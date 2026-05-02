# 228-PINNs-Project Proposal

## Project Title
Solving Fluid Dynamics with Physics-Informed Neural Networks (PINNs)

## Problem Background & Motivation
Traditional numerical methods for solving Partial Differential Equations (PDEs), such as the Navier-Stokes equations for fluid dynamics, often require generating complex meshes and incur massive computational costs. Furthermore, these methods struggle to seamlessly integrate sparse observational data. Physics-Informed Neural Networks (PINNs) offer a paradigm shift by embedding physical laws—expressed as PDEs—directly into the loss function of a neural network. This allows the model to act as a mesh-free PDE solver that is both data-driven and physics-constrained. The motivation of this project is to explore the efficacy and computational efficiency of PINNs in solving complex fluid dynamics problems. By leveraging modern deep learning frameworks (e.g., PyTorch with CUDA Graphs), we aim to demonstrate how PINNs can accelerate the simulation of physical systems compared to traditional numerical solvers, which is highly relevant to the core concepts introduced in Part 2 of this course.

## Related Work
Our project builds upon several key areas within machine learning for physical applications:
*   **Physics-Informed Neural Networks (PINNs):** The foundational work by Raissi et al. (2019) introduced PINNs for solving forward and inverse problems involving nonlinear PDEs. This framework forms the theoretical basis of our project.
*   **Accelerated Deep Learning for Physics:** Recent advancements focus on optimizing PINN training. For instance, the `pinns-torch` framework (Akbarian & Raissi, 2023) utilizes PyTorch's CUDA Graphs and TorchScript to achieve significant speedups over earlier TensorFlow implementations. Our methodology heavily relies on these acceleration techniques.
*   **Neural Operators and Continuous Models:** While Neural ODEs (Chen et al., 2018) and Fourier Neural Operators (FNO) provide alternative continuous-time or operator-learning approaches for physical systems, PINNs remain uniquely suited for mesh-free, equation-driven optimization without requiring massive pre-computed datasets.

## High-level Methodology
We propose to utilize the high-performance `pinns-torch` framework to solve the Navier-Stokes equations for incompressible fluid flow. Our methodology consists of three main phases:
1.  **Problem Formulation & Data Generation:** We will define the spatial-temporal domain and boundary/initial conditions for a classic fluid problem (e.g., cylinder wake). Instead of relying on external datasets, training data (collocation points) will be automatically sampled from the domain using the framework's `MeshSampler`.
2.  **Model Training:** We will construct a Fully Connected Network (FCN) where the outputs (velocity fields $u, v$ and pressure $p$) are constrained by the Navier-Stokes PDE residuals. The loss function will be a weighted sum of the PDE residual loss and the boundary/initial condition loss.
3.  **Evaluation & Optimization:** We will evaluate the model's accuracy against exact solutions or high-fidelity numerical simulations. Furthermore, we will experiment with the framework's CUDA Graphs compilation to measure the training speedup, demonstrating the practical deployment of Physics ML models.