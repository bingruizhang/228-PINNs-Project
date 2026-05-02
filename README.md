# 228-PINNs-Project

## Project Title
Solving Fluid Dynamics with Physics-Informed Neural Networks (PINNs)

## Overview
This project explores the efficacy and computational efficiency of Physics-Informed Neural Networks (PINNs) in solving complex fluid dynamics problems, specifically the Navier-Stokes equations. We utilize the high-performance `pinns-torch` framework to accelerate the simulation of physical systems compared to traditional numerical solvers.

## Project Structure
- `data/`: Contains datasets (will be automatically downloaded by the framework)
- `src/`: Source code for the project
- `docs/`: Documentation and proposal drafts
- `notebooks/`: Jupyter notebooks for data exploration and visualization

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/bingruizhang/228-PINNs-Project.git
   cd 228-PINNs-Project
   ```
2. Install dependencies:
   ```bash
   pip install pinnstorch
   ```
3. Run the Navier-Stokes example:
   ```bash
   python src/train.py
   ```