# Project Proposal: Solving Fluid Dynamics with Physics-Informed Neural Networks (PINNs)

## 1. Problem Background & Motivation

In many engineering and physical science fields, understanding fluid dynamics is very important. Traditionally, people use numerical methods like Computational Fluid Dynamics (CFD) to solve complex Partial Differential Equations (PDEs), such as the Navier-Stokes equations. However, these traditional methods have two main problems: first, they need very complex mesh generation which takes a lot of computational time; second, when we only have a small amount of real-world sensor data, it is hard for traditional solvers to use this sparse data effectively.

Recently, deep learning has shown great power, but standard neural networks (like CNNs or LSTMs) only learn from data and do not "know" the laws of physics. This is where Physics-Informed Neural Networks (PINNs) come in. As we learned in Week 8 of this course, PINNs can embed physical equations directly into the neural network's loss function. This means the network is forced to follow the rules of physics during training. 

Our motivation for this project is to explore how PINNs can act as a "mesh-free" solver for fluid dynamics. We want to see if we can use a modern framework (like `pinns-torch`) to solve the Navier-Stokes equations faster and more easily than traditional methods. Since this course focuses on Machine Learning for Physical Applications (Part 2), we believe studying PINNs is a perfect fit to connect deep learning with real-world physical rules.

## 2. Related Work

Instead of just listing papers, we organize the related research into two main themes that are closely related to our project:

**Theme 1: Deep Learning for Physical Systems and PDEs**
In recent years, researchers have tried to use machine learning to model physical systems. For example, Neural ODEs (which we learned in Week 6) treat hidden layers as continuous time steps to model dynamic systems. Another approach is Neural Operators (like FNO in Week 7), which learn mappings between infinite-dimensional function spaces. While these methods are powerful, they often require a huge amount of pre-computed simulation data to train. In contrast, our project focuses on PINNs, which do not need massive external datasets because they use the PDE itself to guide the learning process.

**Theme 2: The Development and Acceleration of PINNs**
The foundational idea of PINNs was introduced by Raissi et al. (2019), who showed that neural networks can solve forward and inverse PDE problems by adding the PDE residual to the loss function. However, early PINNs built on TensorFlow v1 were sometimes slow to train. Recently, the `pinns-torch` framework (Akbarian & Raissi, 2023) was developed to solve this speed issue. By using PyTorch and CUDA Graphs, it can train models much faster. Our project will build directly on this newer, faster framework to see how well it works in practice.

## 3. High-level Methodology

To achieve our goal, we plan to reproduce and experiment with the Navier-Stokes example using the `pinns-torch` framework. Our methodology is broken down into three main steps:

**Step 1: Problem Setup and Data Generation**
We will focus on a classic 2D fluid problem: the unsteady wake flow behind a cylinder. Instead of downloading gigabytes of external simulation data, we will use the framework's `MeshSampler`. This tool will automatically sample "collocation points" (random points in space and time) and boundary/initial condition points directly from the defined physical domain. 

**Step 2: Model Architecture and Training**
We will build a Fully Connected Network (FCN). The inputs will be spatial coordinates ($x, y$) and time ($t$), and the outputs will be the velocity fields ($u, v$) and pressure ($p$). The core of our method is the loss function. The loss will be calculated by combining two parts: 
1. The Data Loss (how well the model matches the initial and boundary conditions).
2. The Physics Loss (how well the outputs satisfy the Navier-Stokes PDE equations, calculated using automatic differentiation).

**Step 3: Experiments and Evaluation**
First, we will run the baseline model to make sure we can successfully reproduce the fluid flow visualization. After that, we plan to do some simple optimizations. We will change the number of training points (`n_train`) and the number of epochs to see how they affect the final error (MSE) and the training time. 

**Fallback Plan:** If the Navier-Stokes equations are too hard to train on our laptops or take too much time, we will switch to a simpler 1D problem, like the Schrödinger equation or Burgers' equation, which are also supported by the framework. This ensures we can definitely finish the project by the end of the term.