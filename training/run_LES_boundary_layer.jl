"""
Large Eddy Simulation (LES) Runner for Ocean Boundary Layer Training Data Generation

This script runs high-resolution 3D LES simulations of the ocean boundary layer to generate
training data for parameterization development. The LES resolves turbulent eddies explicitly
using fine spatial resolution, providing "ground truth" data for training coarser-scale closures.

Key Features:
- Configurable forcing conditions (momentum, temperature, salinity fluxes)
- Linear initial stratification profiles
- TEOS-10 equation of state for realistic density calculations
- Sponge layer at bottom boundary to prevent reflections
- Checkpoint/restart capability for long simulations
- Comprehensive output: time series of horizontally-averaged profiles and fluxes
- Automatic animation generation

Usage:
    julia run_LES_boundary_layer.jl --QU -2e-5 --QT 2e-4 --stop_time 4 --Nz 128

"""

using Oceananigans
using Oceananigans.Units
using JLD2
using FileIO
using Printf
using CairoMakie
using Oceananigans.Operators
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.BuoyancyModels
using SeawaterPolynomials.TEOS10
using SeawaterPolynomials
using Random
using Statistics
using ArgParse
using LinearAlgebra
using Glob

import Dates

#####
##### Command Line Argument Parsing
#####

"""
Parse command line arguments for simulation configuration.
Returns a dictionary of parameter values.
"""
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        # Surface forcing parameters
        "--QU"
            help = "Surface momentum flux (m²/s²). Negative values represent downward momentum transfer (wind stress)."
            arg_type = Float64
            default = 0.
        "--QT"
            help = "Surface temperature flux (°C m/s). Positive = heating, negative = cooling."
            arg_type = Float64
            default = 0.
        "--QS"
            help = "Surface salinity flux (m/s g/kg). Positive = freshening, negative = salinification."
            arg_type = Float64
            default = 0.

        # Initial conditions
        "--T_surface"
            help = "Surface temperature (°C)"
            arg_type = Float64
            default = 20.
        "--S_surface"
            help = "Surface salinity (g/kg)"
            arg_type = Float64
            default = 35.
        "--dTdz"
            help = "Initial temperature gradient (°C/m). Positive = warming with depth (stabilizing)."
            arg_type = Float64
            default = 1 / 256
        "--dSdz"
            help = "Initial salinity gradient (g/kg/m). Negative = freshening with depth (destabilizing)."
            arg_type = Float64
            default = -0.25 / 256

        # Physical parameters
        "--f"
            help = "Coriolis parameter (s⁻¹). Typical mid-latitude value: 1e-4"
            arg_type = Float64
            default = 1e-4

        # Grid resolution
        "--Nz"
            help = "Number of grid points in z-direction (vertical)"
            arg_type = Int64
            default = 128
        "--Nx"
            help = "Number of grid points in x-direction (horizontal)"
            arg_type = Int64
            default = 256
        "--Ny"
            help = "Number of grid points in y-direction (horizontal)"
            arg_type = Int64
            default = 256

        # Domain dimensions
        "--Lz"
            help = "Domain depth (m)"
            arg_type = Float64
            default = 256.
        "--Lx"
            help = "Domain width in x-direction (m)"
            arg_type = Float64
            default = 512.
        "--Ly"
            help = "Domain width in y-direction (m)"
            arg_type = Float64
            default = 512.

        # Time stepping
        "--dt"
            help = "Initial timestep (seconds)"
            arg_type = Float64
            default = 0.1
        "--max_dt"
            help = "Maximum timestep (minutes)"
            arg_type = Float64
            default = 2.
        "--stop_time"
            help = "Stop time of simulation (days)"
            arg_type = Float64
            default = 4.

        # Output control
        "--time_interval"
            help = "Time interval for time series output (minutes)"
            arg_type = Float64
            default = 10.
        "--field_time_interval"
            help = "Time interval for 3D field output (minutes). Less frequent to save disk space."
            arg_type = Float64
            default = 180.
        "--checkpoint_interval"
            help = "Time interval for checkpoint files (days)"
            arg_type = Float64
            default = 1.
        "--fps"
            help = "Frames per second for animation output"
            arg_type = Float64
            default = 15.

        # Restart control
        "--pickup"
            help = "Whether to pickup from latest checkpoint if available"
            arg_type = Bool
            default = true

        # File I/O
        "--file_location"
            help = "Root directory for output files"
            arg_type = String
            default = "."
    end

    return parse_args(s)
end

# Parse arguments
args = parse_commandline()

# Set random seed for reproducibility of initial noise
Random.seed!(123)

#####
##### Simulation Parameters
#####

# Extract parameters from command line arguments
const Lz = args["Lz"]  # Domain depth (m)
const Lx = args["Lx"]  # Domain width x (m)
const Ly = args["Ly"]  # Domain width y (m)

const Nz = args["Nz"]  # Vertical grid points
const Nx = args["Nx"]  # Horizontal grid points x
const Ny = args["Ny"]  # Horizontal grid points y

const Qᵁ = args["QU"]  # Surface momentum flux (m²/s²)
const Qᵀ = args["QT"]  # Surface temperature flux (°C m/s)
const Qˢ = args["QS"]  # Surface salinity flux (g/kg m/s)

const Pr = 1  # Prandtl number for background diffusivity (ν/κ)

advection = WENO(order=9)
closure = nothing

const f = args["f"]  # Coriolis parameter (s⁻¹)

const dTdz = args["dTdz"]  # Initial temperature gradient (°C/m)
const dSdz = args["dSdz"]  # Initial salinity gradient (g/kg/m)

const T_surface = args["T_surface"]  # Surface temperature (°C)
const S_surface = args["S_surface"]  # Surface salinity (g/kg)

const pickup = args["pickup"]  # Whether to restart from checkpoint

# Use TEOS-10 equation of state for realistic ocean density
const eos = TEOS10EquationOfState()

# Create descriptive filename encoding all simulation parameters
FILE_NAME = "linearTS_dTdz_$(dTdz)_dSdz_$(dSdz)_QU_$(Qᵁ)_QT_$(Qᵀ)_QS_$(Qˢ)_T_$(T_surface)_S_$(S_surface)_f_$(f)_Lxz_$(Lx)_$(Lz)_Nxz_$(Nx)_$(Nz)"
FILE_DIR = joinpath(args["file_location"], FILE_NAME)
mkpath(FILE_DIR)

size_halo = 5
#####
##### Grid Setup
#####

"""
Create a 3D rectilinear grid on GPU.
- Periodic in horizontal (x, y) directions
- Bounded in vertical (z) direction with surface at z=0
"""
grid = RectilinearGrid(GPU(), Float64,
                       size = (Nx, Ny, Nz),
                       halo = (size_halo, size_halo, size_halo),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (-Lz, 0),
                       topology = (Periodic, Periodic, Bounded))

#####
##### Initial Conditions
#####

"""
Add small random noise that decays exponentially with depth.
This seeds turbulence without affecting the large-scale structure.
"""
noise(x, y, z) = rand() * exp(z / 8)

"""
Linear temperature profile: T(z) = T_surface + dTdz * z
For z < 0, this gives T(z) < T_surface if dTdz > 0
"""
T_initial(x, y, z) = dTdz * z + T_surface

"""
Linear salinity profile: S(z) = S_surface + dSdz * z
"""
S_initial(x, y, z) = dSdz * z + S_surface

"""
Add tiny perturbations to seed turbulence
"""
T_initial_noisy(x, y, z) = T_initial(x, y, z) + 1e-6 * noise(x, y, z)
S_initial_noisy(x, y, z) = S_initial(x, y, z) + 1e-6 * noise(x, y, z)

#####
##### Boundary Conditions
#####

"""
Temperature boundary conditions:
- Top: Constant flux Qᵀ (heating/cooling)
- Bottom: Fixed gradient dTdz (maintains stratification)
"""
T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵀ),
                                 bottom=GradientBoundaryCondition(dTdz))

"""
Salinity boundary conditions:
- Top: Constant flux Qˢ (evaporation/precipitation)
- Bottom: Fixed gradient dSdz (maintains stratification)
"""
S_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qˢ),
                                 bottom=GradientBoundaryCondition(dSdz))

"""
Velocity boundary conditions:
- Top: Momentum flux Qᵁ (wind stress in x-direction)
"""
u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵁ))

#####
##### Sponge Layer (Rayleigh Damping)
#####

"""
Apply damping near the bottom boundary to prevent spurious reflections.
This relaxes velocities and tracers back to their initial profiles over
a Gaussian-shaped region centered at the bottom.
"""
damping_rate = 1 / 15minute  # Damping timescale

# Target profiles for relaxation (same as initial conditions)
T_target(x, y, z, t) = T_initial(x, y, z)
S_target(x, y, z, t) = S_initial(x, y, z)

# Gaussian mask centered at bottom with width = Lz/10
bottom_mask = GaussianMask{:z}(center=-grid.Lz, width=grid.Lz/10)

# Create relaxation forcing terms
uvw_sponge = Relaxation(rate=damping_rate, mask=bottom_mask)
T_sponge = Relaxation(rate=damping_rate, mask=bottom_mask, target=T_target)
S_sponge = Relaxation(rate=damping_rate, mask=bottom_mask, target=S_target)

#####
##### Model Construction
#####

"""
Build a NonhydrostaticModel for LES with:
- 3D turbulence-resolving grid
- Coriolis effects (f-plane approximation)
- Realistic seawater equation of state (TEOS-10)
- Temperature and salinity as active tracers
- Sponge layer forcing at bottom
- Surface flux boundary conditions
"""
model = NonhydrostaticModel(
    grid = grid,
    closure = closure,
    coriolis = FPlane(f=f),
    buoyancy = SeawaterBuoyancy(equation_of_state=eos),
    tracers = (:T, :S),
    timestepper = :RungeKutta3,
    advection = advection,
    forcing = (u=uvw_sponge, v=uvw_sponge, w=uvw_sponge, T=T_sponge, S=S_sponge),
    boundary_conditions = (T=T_bcs, S=S_bcs, u=u_bcs)
)

# Extract physical constants for later use
const ρ₀ = eos.reference_density
const g = model.buoyancy.model.gravitational_acceleration

# Set initial conditions
set!(model, T=T_initial_noisy, S=S_initial_noisy)

# Get references to model fields for convenience
T = model.tracers.T
S = model.tracers.S
u, v, w = model.velocities

#####
##### Simulation Setup
#####

"""
Create simulation with adaptive time stepping.
"""
simulation = Simulation(model, Δt=args["dt"]second, stop_time=args["stop_time"]days)

# Adaptive time step wizard with CFL condition
wizard = TimeStepWizard(max_change=1.05, max_Δt=args["max_dt"]minutes, cfl=0.6)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

#####
##### Progress Monitoring
#####

"""
Track wall clock time for performance monitoring
"""
wall_clock = [time_ns()]

"""
Print simulation progress including:
- Current simulation time and iteration
- Wall clock time elapsed
- Maximum velocities and tracer values
- Next timestep size
"""
function print_progress(sim)
    @printf("%s [%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, max(T) %6.3e, max(S) %6.3e, next Δt: %s\n",
            Dates.now(),
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(sim.model.velocities.u),
            maximum(sim.model.velocities.v),
            maximum(sim.model.velocities.w),
            maximum(sim.model.tracers.T),
            maximum(sim.model.tracers.S),
            prettytime(sim.Δt))

    wall_clock[1] = time_ns()
    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))

#####
##### Metadata Saving
#####

"""
Save simulation metadata to output files for reproducibility.
"""
function init_save_some_metadata!(file, model)
    file["metadata/coriolis_parameter"] = f
    file["metadata/momentum_flux"] = Qᵁ
    file["metadata/temperature_flux"] = Qᵀ
    file["metadata/salinity_flux"] = Qˢ
    file["metadata/surface_temperature"] = T_surface
    file["metadata/surface_salinity"] = S_surface
    file["metadata/temperature_gradient"] = dTdz
    file["metadata/salinity_gradient"] = dSdz
    file["metadata/equation_of_state"] = eos
    file["metadata/gravitational_acceleration"] = g
    file["metadata/reference_density"] = ρ₀
    return nothing
end

#####
##### Diagnostic Fields
#####

"""
Compute buoyancy from temperature and salinity using TEOS-10 EOS.
Buoyancy b = -g(ρ - ρ₀)/ρ₀
"""
@inline function get_buoyancy(i, j, k, grid, b, C)
    T, S = Oceananigans.BuoyancyModels.get_temperature_and_salinity(b, C)
    @inbounds ρ = TEOS10.ρ(T[i, j, k], S[i, j, k], 0, eos)
    ρ′ = ρ - ρ₀
    return -g * ρ′ / ρ₀
end

"""
Compute in-situ density from temperature and salinity using TEOS-10 EOS.
"""
@inline function get_density(i, j, k, grid, b, C)
    T, S = Oceananigans.BuoyancyModels.get_temperature_and_salinity(b, C)
    @inbounds ρ = TEOS10.ρ(T[i, j, k], S[i, j, k], 0, eos)
    return ρ
end

# Create computed fields for buoyancy and density
b_op = KernelFunctionOperation{Center, Center, Center}(get_buoyancy, model.grid, model.buoyancy, model.tracers)
b = Field(b_op)
compute!(b)

ρ_op = KernelFunctionOperation{Center, Center, Center}(get_density, model.grid, model.buoyancy, model.tracers)
ρ = Field(ρ_op)
compute!(ρ)

#####
##### Output: Horizontally-Averaged Profiles
#####

"""
Compute horizontal averages (relevant for parameterization training).
These 1D profiles capture the mean state evolution.
"""
ubar = Field(Average(u, dims=(1, 2)))
vbar = Field(Average(v, dims=(1, 2)))
Tbar = Field(Average(T, dims=(1, 2)))
Sbar = Field(Average(S, dims=(1, 2)))
bbar = Field(Average(b, dims=(1, 2)))
ρbar = Field(Average(ρ, dims=(1, 2)))

#####
##### Output: Turbulent Fluxes
#####

"""
Compute resolved turbulent fluxes (horizontally averaged).
These are the key quantities that parameterizations must predict.
- uw, vw: Momentum fluxes
- wT, wS: Temperature and salinity fluxes
- wb: Buoyancy flux
- wρ: Density flux
"""
uw = Field(Average(w * u, dims=(1, 2)))
vw = Field(Average(w * v, dims=(1, 2)))
wb = Field(Average(w * b, dims=(1, 2)))
wT = Field(Average(w * T, dims=(1, 2)))
wS = Field(Average(w * S, dims=(1, 2)))
wρ = Field(Average(w * ρ, dims=(1, 2)))

# Organize outputs
field_outputs = merge(model.velocities, model.tracers)
timeseries_outputs = (; ubar, vbar, Tbar, Sbar, bbar, ρbar,
                        uw, vw, wT, wS, wb, wρ)

#####
##### Output Writers
#####

"""
Set up output writers for different fields.
- 3D instantaneous fields: saved less frequently (large files)
- 1D time series: saved more frequently (small files, key for training)
"""

# Individual velocity components (saved separately to manage file sizes)
simulation.output_writers[:u] = JLD2OutputWriter(model, (; u),
                                                 filename = "$(FILE_DIR)/instantaneous_fields_u.jld2",
                                                 schedule = TimeInterval(args["field_time_interval"]minutes),
                                                 with_halos = true,
                                                 init = init_save_some_metadata!)

simulation.output_writers[:v] = JLD2OutputWriter(model, (; v),
                                                 filename = "$(FILE_DIR)/instantaneous_fields_v.jld2",
                                                 schedule = TimeInterval(args["field_time_interval"]minutes),
                                                 with_halos = true,
                                                 init = init_save_some_metadata!)

simulation.output_writers[:w] = JLD2OutputWriter(model, (; w),
                                                 filename = "$(FILE_DIR)/instantaneous_fields_w.jld2",
                                                 schedule = TimeInterval(args["field_time_interval"]minutes),
                                                 with_halos = true,
                                                 init = init_save_some_metadata!)

simulation.output_writers[:T] = JLD2OutputWriter(model, (; T),
                                                 filename = "$(FILE_DIR)/instantaneous_fields_T.jld2",
                                                 schedule = TimeInterval(args["field_time_interval"]minutes),
                                                 with_halos = true,
                                                 init = init_save_some_metadata!)

simulation.output_writers[:S] = JLD2OutputWriter(model, (; S),
                                                 filename = "$(FILE_DIR)/instantaneous_fields_S.jld2",
                                                 schedule = TimeInterval(args["field_time_interval"]minutes),
                                                 with_halos = true,
                                                 init = init_save_some_metadata!)

simulation.output_writers[:b] = JLD2OutputWriter(model, (; b),
                                                 filename = "$(FILE_DIR)/instantaneous_fields_b.jld2",
                                                 schedule = TimeInterval(args["field_time_interval"]minutes),
                                                 with_halos = true,
                                                 init = init_save_some_metadata!)

simulation.output_writers[:ρ] = JLD2OutputWriter(model, (; ρ),
                                                 filename = "$(FILE_DIR)/instantaneous_fields_rho.jld2",
                                                 schedule = TimeInterval(args["field_time_interval"]minutes),
                                                 with_halos = true,
                                                 init = init_save_some_metadata!)

# Time series of horizontal averages and fluxes (most important for training)
simulation.output_writers[:timeseries] = JLD2OutputWriter(model, timeseries_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_timeseries.jld2",
                                                          schedule = TimeInterval(args["time_interval"]minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)

# Checkpointer for restart capability
simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule=TimeInterval(args["checkpoint_interval"]days),
                                                        prefix="$(FILE_DIR)/model_checkpoint")

#####
##### Run Simulation
#####

"""
Run the simulation, optionally picking up from the latest checkpoint.
"""
if pickup
    files = readdir(FILE_DIR)
    checkpoint_files = files[occursin.(Ref("model_checkpoint_iteration"), files)]
    if !isempty(checkpoint_files)
        # Extract iteration numbers from checkpoint filenames
        checkpoint_iters = parse.(Int, [filename[findfirst("iteration", filename)[end]+1:findfirst(".jld2", filename)[1]-1] for filename in checkpoint_files])
        pickup_iter = maximum(checkpoint_iters)
        @info "Picking up from checkpoint at iteration $(pickup_iter)"
        run!(simulation, pickup="$(FILE_DIR)/model_checkpoint_iteration$(pickup_iter).jld2")
    else
        @info "No checkpoint found, starting from initial conditions"
        run!(simulation)
    end
else
    @info "Starting simulation from initial conditions"
    run!(simulation)
end

# Clean up checkpoint files after successful completion
checkpointers = glob("$(FILE_DIR)/model_checkpoint_iteration*.jld2")
if !isempty(checkpointers)
    @info "Removing checkpoint files..."
    rm.(checkpointers)
end

@info "Simulation completed successfully!"

#####
##### Post-Processing: Load Data for Visualization
#####

@info "Loading data for visualization..."

# Load time series data
ubar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "ubar")
vbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vbar")
Tbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "Tbar")
Sbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "Sbar")
bbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar")
ρbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "ρbar")

uw_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "uw")
vw_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vw")
wT_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wT")
wS_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wS")
wb_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wb")
wρ_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wρ")

# Extract grid information
xC = Tbar_data.grid.xᶜᵃᵃ[1:Nx]
yC = Tbar_data.grid.yᵃᶜᵃ[1:Ny]
zC = Tbar_data.grid.zᵃᵃᶜ[1:Nz]  # Cell centers
zF = uw_data.grid.zᵃᵃᶠ[1:Nz+1]  # Cell faces (for fluxes)

Nt = length(Tbar_data.times)

#####
##### Visualization: Animated Time Series
#####

@info "Creating animation..."

# Create figure with multiple subplots
fig = Figure(size=(2500, 1800))

# Mean profiles (top row)
axTbar = Axis(fig[1:2, 1:2], title="⟨T⟩ Temperature", xlabel="°C", ylabel="z (m)")
axSbar = Axis(fig[1:2, 3:4], title="⟨S⟩ Salinity", xlabel="g kg⁻¹", ylabel="z (m)")
axubar = Axis(fig[1, 5], title="⟨u⟩ Velocity", xlabel="m s⁻¹", ylabel="z (m)")
axvbar = Axis(fig[1, 6], title="⟨v⟩ Velocity", xlabel="m s⁻¹", ylabel="z (m)")
axbbar = Axis(fig[2, 5], title="⟨b⟩ Buoyancy", xlabel="m s⁻²", ylabel="z (m)")
axρbar = Axis(fig[2, 6], title="⟨ρ⟩ Density", xlabel="kg m⁻³", ylabel="z (m)")

# Turbulent fluxes (bottom row)
axwT = Axis(fig[3:4, 1:2], title="⟨wT⟩ Heat Flux", xlabel="m s⁻¹ °C", ylabel="z (m)")
axwS = Axis(fig[3:4, 3:4], title="⟨wS⟩ Salt Flux", xlabel="m s⁻¹ g kg⁻¹", ylabel="z (m)")
axuw = Axis(fig[3, 5], title="⟨uw⟩ Momentum Flux", xlabel="m² s⁻²", ylabel="z (m)")
axvw = Axis(fig[3, 6], title="⟨vw⟩ Momentum Flux", xlabel="m² s⁻²", ylabel="z (m)")
axwb = Axis(fig[4, 5], title="⟨wb⟩ Buoyancy Flux", xlabel="m² s⁻³", ylabel="z (m)")
axwρ = Axis(fig[4, 6], title="⟨wρ⟩ Density Flux", xlabel="kg m⁻² s⁻¹", ylabel="z (m)")

# Compute axis limits from data
ubarlim = (minimum(ubar_data), maximum(ubar_data))
vbarlim = (minimum(vbar_data), maximum(vbar_data))
Tbarlim = (minimum(Tbar_data) - 1e-3, maximum(Tbar_data) + 1e-3)
Sbarlim = (minimum(Sbar_data) - 1e-3, maximum(Sbar_data) + 1e-3)
bbarlim = (minimum(bbar_data), maximum(bbar_data))
ρbarlim = (minimum(ρbar_data), maximum(ρbar_data))

# For fluxes, skip early transient spinup in computing limits
startframe_lim = 30
uwlim = (minimum(uw_data[1, 1, :, startframe_lim:end]), maximum(uw_data[1, 1, :, startframe_lim:end]))
vwlim = (minimum(vw_data[1, 1, :, startframe_lim:end]), maximum(vw_data[1, 1, :, startframe_lim:end]))
wTlim = (minimum(wT_data[1, 1, :, startframe_lim:end]), maximum(wT_data[1, 1, :, startframe_lim:end]))
wSlim = (minimum(wS_data[1, 1, :, startframe_lim:end]), maximum(wS_data[1, 1, :, startframe_lim:end]))
wblim = (minimum(wb_data[1, 1, :, startframe_lim:end]), maximum(wb_data[1, 1, :, startframe_lim:end]))
wρlim = (minimum(wρ_data[1, 1, :, startframe_lim:end]), maximum(wρ_data[1, 1, :, startframe_lim:end]))

# Observable for animation frame number
n = Observable(1)

# Title with simulation parameters and current time
time_str = @lift "Qᵁ = $(Qᵁ), Qᵀ = $(Qᵀ), Qˢ = $(Qˢ), f = $(f), Time = $(round(Tbar_data.times[$n]/24/60^2, digits=3)) days"
title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

# Extract data for current frame
ubarₙ = @lift interior(ubar_data[$n], 1, 1, :)
vbarₙ = @lift interior(vbar_data[$n], 1, 1, :)
Tbarₙ = @lift interior(Tbar_data[$n], 1, 1, :)
Sbarₙ = @lift interior(Sbar_data[$n], 1, 1, :)
bbarₙ = @lift interior(bbar_data[$n], 1, 1, :)
ρbarₙ = @lift interior(ρbar_data[$n], 1, 1, :)

uwₙ = @lift interior(uw_data[$n], 1, 1, :)
vwₙ = @lift interior(vw_data[$n], 1, 1, :)
wTₙ = @lift interior(wT_data[$n], 1, 1, :)
wSₙ = @lift interior(wS_data[$n], 1, 1, :)
wbₙ = @lift interior(wb_data[$n], 1, 1, :)
wρₙ = @lift interior(wρ_data[$n], 1, 1, :)

# Plot profiles (will update automatically via Observables)
lines!(axubar, ubarₙ, zC)
lines!(axvbar, vbarₙ, zC)
lines!(axTbar, Tbarₙ, zC)
lines!(axSbar, Sbarₙ, zC)
lines!(axbbar, bbarₙ, zC)
lines!(axρbar, ρbarₙ, zC)

lines!(axuw, uwₙ, zF)
lines!(axvw, vwₙ, zF)
lines!(axwT, wTₙ, zF)
lines!(axwS, wSₙ, zF)
lines!(axwb, wbₙ, zF)
lines!(axwρ, wρₙ, zF)

# Set fixed axis limits
xlims!(axubar, ubarlim)
xlims!(axvbar, vbarlim)
xlims!(axTbar, Tbarlim)
xlims!(axSbar, Sbarlim)
xlims!(axbbar, bbarlim)
xlims!(axρbar, ρbarlim)

xlims!(axuw, uwlim)
xlims!(axvw, vwlim)
xlims!(axwT, wTlim)
xlims!(axwS, wSlim)
xlims!(axwb, wblim)
xlims!(axwρ, wρlim)

trim!(fig.layout)
# display(fig)

# Record animation
@info "Rendering video..."

CairoMakie.record(fig, joinpath(FILE_DIR, "$(FILE_NAME)_timeseries.mp4"), 1:Nt, framerate=15) do nn
    n[] = nn
end

@info "Animation saved to $(FILE_DIR)/$(FILE_NAME)_timeseries.mp4"
@info "All processing completed!"
