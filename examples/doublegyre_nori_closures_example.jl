# Example: Using NORi Closures in a Double Gyre Simulation
#
# This script demonstrates the correct usage of NORiBaseVerticalDiffusivity
# and NORiNNFluxClosure in an Oceananigans simulation using a double gyre
# configuration with realistic ocean forcing.
#
# Key points:
# 1. NORiBaseVerticalDiffusivity must be first in the closure tuple
# 2. NORiNNFluxClosure must be second (immediately after base)
# 3. Use helper functions to ensure correct ordering

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode
using SeawaterPolynomials
using SeawaterPolynomials: TEOS10
using Printf
using Statistics
import Dates

# Import NORi closures from Implementation module
using NORiOceanParameterization.Implementation

#####
##### Simulation setup
#####

model_architecture = GPU()

#####
##### Closure setup - METHOD 1: Using helper function (RECOMMENDED)
#####
# This is the recommended approach - the helper function ensures correct ordering
closure = NORiClosureWithNN(arch=model_architecture)

# Alternative: If you only want the base closure without NN
# closure = NORiBaseClosureOnly()

# Alternative: Mix with other Oceananigans closures
# using Oceananigans.TurbulenceClosures
# scalar_diff = ScalarDiffusivity(ν=1e-2, κ=1e-2)  # Horizontal diffusion
# closure = NORiClosureTuple(:base, :nn, scalar_diff, arch=model_architecture)

@info "Model Closure: $closure"
#####
##### Grid setup
#####
# Grid parameters
const Nx = 100
const Ny = 100
const Nz = 200

const Δz = 8meters
const Lx = 4000kilometers
const Ly = 6000kilometers
const Lz = Nz * Δz

grid = RectilinearGrid(model_architecture, Float64,
                       topology = (Bounded, Bounded, Bounded),
                       size = (Nx, Ny, Nz),
                       halo = (4, 4, 4),
                       x = (-Lx/2, Lx/2),
                       y = (-Ly/2, Ly/2),
                       z = (-Lz, 0))

advection_scheme = WENO(grid, order = 5)

@info "Built a grid: $grid."

#####
##### Boundary conditions
#####

const T_north = 0
const T_south = 30
const T_mid = (T_north + T_south) / 2
const ΔT = T_south - T_north

const S_north = 34
const S_south = 37
const S_mid = (S_north + S_south) / 2

const τ₀ = 1e-4

const μ_drag = 1/30days
const μ_T = 1/8days

#####
##### Forcing and initial conditions
#####

@inline T_initial(x, y, z) = 10 + 20 * (1 + z / Lz)
@inline surface_u_flux(x, y, t) = -τ₀ * cos(2π * y / Ly)

surface_u_flux_bc = FluxBoundaryCondition(surface_u_flux)

@inline u_drag(x, y, t, u) = @inbounds -μ_drag * Lz * u
@inline v_drag(x, y, t, v) = @inbounds -μ_drag * Lz * v

u_drag_bc = FluxBoundaryCondition(u_drag; field_dependencies=:u)
v_drag_bc = FluxBoundaryCondition(v_drag; field_dependencies=:v)

u_bcs = FieldBoundaryConditions(top = surface_u_flux_bc,
                                bottom = u_drag_bc,
                                north = ValueBoundaryCondition(0),
                                south = ValueBoundaryCondition(0))

v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(0),
                                bottom = v_drag_bc,
                                east = ValueBoundaryCondition(0),
                                west = ValueBoundaryCondition(0))

@inline T_ref(y) = T_mid - ΔT / Ly * y
@inline surface_T_flux(x, y, t, T) = μ_T * Δz * (T - T_ref(y))
surface_T_flux_bc = FluxBoundaryCondition(surface_T_flux; field_dependencies=:T)
T_bcs = FieldBoundaryConditions(top = surface_T_flux_bc)

@inline S_ref(y) = (S_north - S_south) / Ly * y + S_mid
@inline S_initial(x, y, z) = S_ref(y)
@inline surface_S_flux(x, y, t, S) = μ_T * Δz * (S - S_ref(y))
surface_S_flux_bc = FluxBoundaryCondition(surface_S_flux; field_dependencies=:S)
S_bcs = FieldBoundaryConditions(top = surface_S_flux_bc)

#####
##### Coriolis
#####

coriolis = BetaPlane(rotation_rate=7.292115e-5, latitude=45, radius=6371e3)

#####
##### Model building
#####

# This is a weird bug. If a model is not initialized with a closure other than NORi,
# the code will throw a CUDA: illegal memory access error for models larger than a small size.
# This is a workaround to initialize the model with a closure other than NORi first,
# then the code will run without any issues.
model = HydrostaticFreeSurfaceModel(
    grid = grid,
    free_surface = SplitExplicitFreeSurface(grid, cfl=0.75),
    momentum_advection = advection_scheme,
    tracer_advection = advection_scheme,
    buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10.TEOS10EquationOfState()),
    coriolis = coriolis,
    closure = VerticalScalarDiffusivity(ν=1e-5, κ=1e-5),
    tracers = (:T, :S),
    boundary_conditions = (; u = u_bcs, v = v_bcs, T = T_bcs, S = S_bcs),
)

@info "Building a model with NORi closures..."

model = HydrostaticFreeSurfaceModel(;
    grid = grid,
    free_surface = SplitExplicitFreeSurface(grid, cfl=0.75),
    momentum_advection = advection_scheme,
    tracer_advection = advection_scheme,
    buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10.TEOS10EquationOfState()),
    coriolis = coriolis,
    closure = closure,
    tracers = (:T, :S),
    boundary_conditions = (; u = u_bcs, v = v_bcs, T = T_bcs, S = S_bcs),
)

@info "Built $model."

#####
##### Initial conditions
#####

noise(z) = rand() * exp(z / 8)

T_initial_noisy(x, y, z) = T_initial(x, y, z) + 1e-6 * noise(z)
S_initial_noisy(x, y, z) = S_initial(x, y, z) + 1e-6 * noise(z)

set!(model, T=T_initial_noisy, S=S_initial_noisy)

using Oceananigans.TimeSteppers: update_state!
update_state!(model)

#####
##### Simulation building
#####

Δt₀ = 5minutes
stop_time = 100 * 365days

simulation = Simulation(model, Δt = Δt₀, stop_time = stop_time)

#####
##### Callbacks
#####

# Progress callback
wall_clock = [time_ns()]

function print_progress(sim)
    @printf("%s, [%05.2f%%] i: %d, t: %s, wall time: %s, max(u): %6.3e, max(v): %6.3e, max(T): %6.3e, max(S): %6.3e, next Δt: %s\n",
        Dates.now(),
        100 * (sim.model.clock.time / sim.stop_time),
        sim.model.clock.iteration,
        prettytime(sim.model.clock.time),
        prettytime(1e-9 * (time_ns() - wall_clock[1])),
        maximum(abs, sim.model.velocities.u),
        maximum(abs, sim.model.velocities.v),
        maximum(abs, sim.model.tracers.T),
        maximum(abs, sim.model.tracers.S),
        prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(1))

run!(simulation)

@info "Simulation complete!"