# NORiOceanParameterization.jl

[![Julia](https://img.shields.io/badge/julia-v1.10+-blue.svg)](https://julialang.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Neural ODEs ("NO") and Richardson Number ("Ri")-Based Ocean Parameterization**

A ML-augmented ocean turbulence parameterization that combines physics-based Richardson number closures with neural networks to improve vertical mixing predictions in realistic ocean models. Designed for integration with [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl). 

Trained entirely end-to-end (_a posteriori_) with fully-differentiable ODE solvers and Ensemble Kalman Inversion.

## Overview

NORiOceanParameterization.jl provides hybrid ocean parameterizations that:

- **Use physics-based Richardson number closures** to model vertical diffusivity in different turbulent regimes (convective vs. shear-driven)
- **Apply neural networks** to learn entrainment processes during convection, acting as a residual flux
- **Provide an end-to-end training framework** using fully differentiable-(Neural ODEs) and ensemble-based (Ensemble Kalman Inversion) methods
- **Provide calibrated closures** trained on large-eddy simulations (LES) generated with Oceananigans
- **Integrate seamlessly with Oceananigans.jl** for use in ocean simulations on CPUs and GPUs

## Repository Structure

The repository is organized into several directories:

### Core Implementation
- **[`src/`](src/)**: Core package source code
  - `Closures/`: Closure implementations (base Richardson number closure)
  - `Implementation/`: Oceananigans.jl integration and convenience functions
  - `DataWrangling/`: Tools for loading and processing data
  - `ODEOperations/`: ODE solvers and operations used during training and inference
  - `Operators/`: Differential operators for the parameterizations
  - `TrainingOperations/`: Training utilities, neural differential equation and loss function definition
  - `Plotting/`: Visualization utilities
  - `Utils/`: General utility functions

### Pre-Trained Models
- **[`calibrated_parameters/`](calibrated_parameters/)**: Pre-trained parameters for base Richardson number closure and final neural network weights

### Training and Validation
- **[`training/`](training/)**: Complete training pipeline scripts
  - `run_LES_boundary_layer.jl`: Generate LES training data with Oceananigans.jl
  - `train_NDE.jl`: Neural Differential Equations training for NN closure
  - `train_localbaseclosure_convection.jl`: EKI calibration for convective regime of the base closure
  - `train_localbaseclosure_shear.jl`: EKI calibration for shear regime of the base closure

### Inference and Examples
- **[`inference/`](inference/)**: Example scripts for using NORi closures in Oceananigans models
  - `column_model_nori_closures_example.jl`: 1D column model example
  - `doublegyre_nori_closures_example.jl`: 3D double-gyre circulation example

- **[`experiments/`](experiments/)**: Validation and benchmark experiments
  - Column model simulations comparing NORi against LES, CATKE, and k-epsilon closures
  - Scripts for timestep dependence studies and long integration tests

### Visualization
- **[`figure_scripts/`](figure_scripts/)**: Scripts to generate all publication figures
  - Training results visualization (NDE losses, training metrics)
  - Validation plots comparing NORi against LES, CATKE, and k-epsilon closures
  - 3D LES field visualizations
  - Double-gyre circulation analysis plots
  - Data required to plot the figures are hosted on this [Zenodo data companion](https://doi.org/10.5281/zenodo.17605195)
  - Data loading from Zenodo is handled automatically with DataDeps.jl

## Key Features

### Hybrid Physics-ML Approach

#### `NORiBaseVerticalDiffusivity`: A Richardson number-based vertical diffusivity closure with
- Separate parameterizations for convective (Ri < 0) and shear (Ri > 0) regimes
- Distinct Prandtl numbers for convective and shear regimes
- **Pre-trained parameters** loaded automatically from calibration

#### `NORiNNFluxClosure`: Neural network closure for residual flux corrections with
- Separate networks for temperature and salinity fluxes
- **21 input features**: Richardson number, temperature/salinity/density gradients at 5 vertical levels, surface buoyancy flux
- Only active near base of the mixed layer (where entrainment mixing happens)
- **Pre-trained Lux.jl models** included

This hybrid approach combines interpretable physical closures with highly-expressive neural networks `NORiBaseVerticalDiffusivity` + `NORiNNFluxClosure`:

### Designed for Realistic Ocean Equation of State (TEOS-10)
NORi is specifically trained on and designed for the nonlinear equation of state based on TEOS-10 [(Roquet et al., 2015)](https://doi.org/10.1016/j.ocemod.2015.04.002), as implemented in [SeawaterPolynomials.jl](https://github.com/CliMA/SeawaterPolynomials.jl). This becomes important in cold regions (e.g. Southern Ocean) where the seawater's equation of state is highly nonlinear.

### Complete ML Training Pipeline

The repository includes a full training infrastructure:
- **Neural Differential Equations (NDEs)** for fully-differentiable neural network calibration over ODE solvers
- **Ensemble Kalman Inversion (EKI)** for base closure calibration 
- **Free and Open Source** training dataset [SOBLLES: Salty Ocean Boundary Layer Large-Eddy Simulations](https://doi.org/10.5281/zenodo.16278000) covering a wide range of realistic ocean scenarios
- Cutting-edge automatic differentiation via [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)
- Highly efficient gradient-free methods using [EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl)
- Multi-initial condition training with adaptive loss weighting

## Important Caveats (for now!)
### Package Compatibility
Due to rapid software development cycles with potentially breaking changes, we require
- Julia 1.10
- Enzyme v0.11.20
- Lux v0.5.13

Using other versions may lead to potential software issues, especially for the end-to-end training with Enzyme `autodiff`.

### Using NORi in production
NORi can be readily used in the `HydrostaticFreeSurfaceModel` Oceananigans. Details and examples can be found in the [`src/Implementation`](src/Implementation) and [`inference`](inference) directories. However, it currently only supports a grid with uniform vertical spacing of ` Δz = 8 m`. Support for nonuniform grids in the vertical will be coming soon!

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/xkykai/NORiOceanParameterization.jl")
```

## Quick Start

### Using Pre-Trained Closures

```julia
using NORiOceanParameterization.Implementation
using Oceananigans
using Oceananigans.Units

model_architecture = CPU()
# To use GPUs, uncomment the following instead:
# using CUDA
# model_architecture = CPU()

# Create closures with trained parameters (automatically loaded)
closures = NORiClosureWithNN(arch=model_architecture)

# Set up a simple ocean column
grid = RectilinearGrid(size = (32, 32, 32),
                       x = (0, 1),
                       y = (0, 1),
                       z = (-128, 0),
                       topology = (Periodic, Periodic, Bounded))

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    closure = closures,
                                    tracers = (:T, :S),
                                    buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10.TEOS10EquationOfState()))

# Run simulation
simulation = Simulation(model, Δt=5minutes, stop_time=1day)
run!(simulation)
```

## Training Your Own Closures

See training scripts in [`training/`](training/) for complete training pipeline examples!

<!-- ## Citation

If you use this package in your research, please cite:

```bibtex
@software{NORiOceanParameterization,
  author = {Lee, Xin Kai},
  title = {NORiOceanParameterization.jl: Neural ODEs and Richardson Number-Based Ocean Parameterization},
  year = {2025},
  url = {https://github.com/xkykai/NORiOceanParameterization.jl}
}
``` -->

## License

This project is licensed under the MIT License.