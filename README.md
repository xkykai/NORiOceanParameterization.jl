# NORiOceanParameterization.jl

[![Julia](https://img.shields.io/badge/julia-v1.10+-blue.svg)](https://julialang.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/1092643359.svg)](https://doi.org/10.5281/zenodo.17773380)

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
- **Validated against real ocean observations** at Ocean Station Papa (NOAA PMEL surface mooring) under both prescribed-atmosphere (bulk formulae) and prescribed-flux boundary conditions

## Repository Structure

The repository is organized into several directories:

### Core Package
- **[`src/`](src/)**: Core package source code for training and data processing
  - `Closures/`: Base closure implementations used during training
  - `DataWrangling/`: Tools for loading and processing LES data
  - `ODEOperations/`: ODE solvers and operations used during training
  - `Operators/`: Differential operators for the parameterizations
  - `TrainingOperations/`: Training utilities, neural differential equation and loss function definition
  - `Plotting/`: Visualization utilities
  - `Utils/`: Model I/O utilities (saving/loading NN weights and parameters)

### Inference Package
- **[`NORiImplementation/`](NORiImplementation/)**: Standalone package for using trained closures in Oceananigans
  - `src/NORiBaseVerticalDiffusivity.jl`: Richardson number-based vertical diffusivity closure
  - `src/NORiNNFluxClosure.jl`: Neural network closure for residual flux corrections
  - `src/closure_builder.jl`: Helper functions for combining closures correctly
  - Uses Oceananigans v0.106, Lux (v1.x), and depends on `NumericalEarth` and `OceanStationPapa`

### Ocean Station Papa
- **[`OceanStationPapa/`](OceanStationPapa/)**: Standalone package for forcing single-column ocean models with real observations from the NOAA PMEL Ocean Station Papa mooring (50.1°N, 144.9°W)
  - Auto-downloads the hourly mooring NetCDF from the NOAA S3 bucket (cached via `Scratch.jl`)
  - Exposes the data through the `NumericalEarth.DataWrangling` interface (`OceanStationPapaHourly`, `OceanStationPapaFluxHourly`) for use as initial conditions
  - Provides `OceanStationPapaPrescribedAtmosphere` for bulk-formula forcing, and `ocean_station_papa_prescribed_fluxes` / `ocean_station_papa_prescribed_flux_boundary_conditions` for direct surface-flux forcing
  - Handles unit conversion (Celsius → Kelvin, millibar → Pa, cm/s → m/s, mm/hr → m/s) and gap-filling of missing hourly samples
  - Depends on `NumericalEarth` v0.2.3, `Oceananigans` v0.106, `Thermodynamics`, and `ClimaSeaIce`

### Pre-Trained Models
- **[`calibrated_parameters/`](calibrated_parameters/)**: Pre-trained parameters for base Richardson number closure and final neural network weights

### Training and Validation
- **[`training/`](training/)**: Complete training pipeline scripts
  - `run_LES_boundary_layer.jl`: Generate LES training data with Oceananigans.jl
  - `train_NDE.jl`: Neural Differential Equations training for NN closure
  - `train_localbaseclosure_convection.jl`: EKI calibration for convective regime of the base closure
  - `train_localbaseclosure_shear.jl`: EKI calibration for shear regime of the base closure
  - Pins old dependencies (Enzyme v0.11.20, Lux v0.5.31) for compatibility

### Inference and Examples
- **[`inference/`](inference/)**: Example scripts for using NORi closures in Oceananigans models
  - `column_model_nori_closures_example.jl`: 1D column model with idealized forcing
  - `doublegyre_nori_closures_example.jl`: 3D double-gyre circulation
  - `column_model_ospapa_prescribedatmosphere.jl`: OS Papa single-column with `PrescribedAtmosphere` + bulk formulae
  - `column_model_ospapa_prescribedfluxes.jl`: OS Papa single-column with observed surface fluxes as Oceananigans `FluxBoundaryCondition`s
  - `column_model_ospapa_prescribedfluxes_conservative.jl`: as above, with a per-year global flux correction to close the heat/salt budget

- **[`experiments/`](experiments/)**: Validation and benchmark experiments
  - Column model simulations comparing NORi against LES, CATKE, and k-epsilon closures
  - Scripts for timestep dependence studies and long integration tests

### Visualization
- **[`figure_scripts/`](figure_scripts/)**: Scripts to generate all publication figures (a self-contained Julia project)
  - Training results visualization (NDE losses, training metrics)
  - Validation plots comparing NORi against LES, CATKE, and k-epsilon closures
  - 3D LES field visualizations
  - Double-gyre circulation analysis plots
  - **[`figure_scripts/ospapa/`](figure_scripts/ospapa/)**: subproject for OS Papa drift and climatology comparison plots against CATKE and k-epsilon
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

### Validated Against Real Ocean Observations at Ocean Station Papa
NORi is exercised on continuous, multi-year observations from the NOAA PMEL Ocean Station Papa mooring (50.1°N, 144.9°W), which has been recording near-surface temperature, salinity, velocity, and atmospheric state since 2007. The repository supports two complementary single-column configurations:

- **Prescribed atmosphere**: observed atmospheric state is fed into `NumericalEarth.Atmospheres.PrescribedAtmosphere`, and surface fluxes are computed inside Oceananigans via bulk formulae (see [`inference/column_model_ospapa_prescribedatmosphere.jl`](inference/column_model_ospapa_prescribedatmosphere.jl))
- **Prescribed surface fluxes**: observed (or bulk-derived) air-sea fluxes are applied directly as `FluxBoundaryCondition`s, with an optional per-year global correction for budget closure (see [`inference/column_model_ospapa_prescribedfluxes.jl`](inference/column_model_ospapa_prescribedfluxes.jl) and [`inference/column_model_ospapa_prescribedfluxes_conservative.jl`](inference/column_model_ospapa_prescribedfluxes_conservative.jl))

Drift and climatology plots comparing NORi against CATKE and k-epsilon live in [`figure_scripts/ospapa/`](figure_scripts/ospapa/). The mooring NetCDF is auto-downloaded from the NOAA S3 bucket on first use by [`OceanStationPapa/`](OceanStationPapa/).

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

This repository is organized into separate environments to handle conflicting dependency versions:

- **`NORiImplementation/`** — For running pre-trained closures in Oceananigans (latest Lux v1.x, Oceananigans v0.106; depends on `NumericalEarth` and `OceanStationPapa`)
- **`OceanStationPapa/`** — Observational data and forcing helpers for Ocean Station Papa (Oceananigans v0.106, NumericalEarth v0.2.3)
- **`training/`** — For training new closures (pins Enzyme v0.11.20 and Lux v0.5.31)

Using other versions may lead to potential software issues, especially for the end-to-end training with Enzyme `autodiff`. The upstream churn in Oceananigans between v0.104 and v0.106 (boundary condition and closure interface changes) is the reason `NORiImplementation/` and `OceanStationPapa/` are pinned to v0.106.

### Using NORi in production
NORi can be readily used in the `HydrostaticFreeSurfaceModel` in Oceananigans. Details and examples can be found in the [`NORiImplementation/`](NORiImplementation) and [`inference/`](inference) directories. However, it currently only supports a grid with uniform vertical spacing of `Δz = 8 m`. Support for nonuniform grids in the vertical will be coming soon!

## Installation

### For inference (using pre-trained closures)

From the repository root:
```bash
julia --project=NORiImplementation -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

For the Ocean Station Papa examples, also dev-link the `OceanStationPapa` subpackage:
```bash
julia --project=NORiImplementation -e 'using Pkg; Pkg.develop([Pkg.PackageSpec(path="."), Pkg.PackageSpec(path="OceanStationPapa")]); Pkg.instantiate()'
```

### For training

The training scripts auto-bootstrap the environment on first run. From the repository root:

```bash
julia training/train_NDE.jl
```

Or set up manually:

```bash
julia --project=training -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

## Quick Start

### Using Pre-Trained Closures

```julia
using NORiImplementation
using Oceananigans
using Oceananigans.Units

# Create closures with trained parameters (automatically loaded)
closures = NORiClosureWithNN()  # For CPU

# For GPU, use:
# using CUDA
# closures = NORiClosureWithNN(arch=GPU())

# Set up a simple ocean column
grid = RectilinearGrid(size = (32, 32, 32),
                       x = (0, 1),
                       y = (0, 1),
                       z = (-128, 0),
                       topology = (Periodic, Periodic, Bounded))

model = HydrostaticFreeSurfaceModel(grid;
                                    closure = closures,
                                    tracers = (:T, :S),
                                    buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10.TEOS10EquationOfState()))

# Run simulation
simulation = Simulation(model, Δt=5minutes, stop_time=1day)
run!(simulation)
```

### Running a NORi Column at Ocean Station Papa

A minimal sketch using observed atmospheric forcing (the full example, including initial conditions, year sweep, and progress callback, lives in [`inference/column_model_ospapa_prescribedatmosphere.jl`](inference/column_model_ospapa_prescribedatmosphere.jl)):

```julia
using NumericalEarth
using OceanStationPapa
using Oceananigans
using Oceananigans.Units
using NORiImplementation
using Dates

arch = CPU()
start_date = DateTime(2018, 10, 1)
end_date   = start_date + Day(150)

# Single-column grid at the OS Papa location
grid = RectilinearGrid(arch, size = 25,
                       x = -144.9, y = 50.1, z = (-200, 0),
                       topology = (Flat, Flat, Bounded))

ocean = ocean_simulation(grid;
                         Δt        = 10minutes,
                         coriolis  = FPlane(latitude=50.1),
                         closure   = NORiClosureWithNN(arch=arch))

# Initialize T, S from the OS Papa mooring profile
set!(ocean.model, T=Metadatum(:temperature, dataset=OceanStationPapaHourly(), date=start_date),
                  S=Metadatum(:salinity,    dataset=OceanStationPapaHourly(), date=start_date))

# Force with the observed atmospheric state (bulk formulae compute fluxes internally)
atmosphere = OceanStationPapaPrescribedAtmosphere(; start_date, end_date, surface_layer_height = 2.5)
coupled    = OceanOnlyModel(ocean; atmosphere, radiation = Radiation())
simulation = Simulation(coupled, Δt=ocean.Δt, stop_time=atmosphere.times[end])
run!(simulation)
```

## Training Your Own Closures

Training scripts live in [`training/`](training/) and use a separate environment with pinned dependency versions (Enzyme v0.11.20, Lux v0.5.31). Each script auto-activates this environment and dev-installs the core package on first run.

```bash
# Run NDE training
julia training/train_NDE.jl

# Or run with explicit project activation
julia --project=training training/train_NDE.jl
```

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